"""
Custom reward function for Search-R1, compatible with upstream verl 0.7.1.
verl calls: compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
"""

import re
import string


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
    if len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    if em_check(answer, ground_truth['target']):
        return 1.0
    return 0.0
