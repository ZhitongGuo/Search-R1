"""
Standalone reward function for Search-R1 training.
Extracts answer from <answer>...</answer> tags and checks exact match.
"""

import re
import string
import torch
from verl import DataProto


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized = normalize_answer(prediction)
    return any(normalize_answer(g) == normalized for g in golden_answers)


def extract_answer(solution_str):
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if len(matches) <= 1:
        return None
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, format_score=0., score=1.):
    answer = extract_answer(solution_str)
    if answer is None:
        return 0
    return score if em_check(answer, ground_truth['target']) else format_score


class RewardManager:
    def __init__(self, tokenizer, num_examine=0, format_score=0.):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        printed = {}

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = item.batch['responses']
            valid_response_length = item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences_str = self.tokenizer.decode(torch.cat((valid_prompt_ids, valid_response_ids)))
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            data_source = item.non_tensor_batch['data_source']

            score = compute_score_em(sequences_str, ground_truth, format_score=self.format_score)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in printed:
                printed[data_source] = 0
            if printed[data_source] < self.num_examine:
                printed[data_source] += 1
                print(sequences_str)

        return reward_tensor
