"""
Custom AgentLoopManager for Search-R1 that batches search queries across all samples.

Instead of processing tool calls per-sample (which causes N individual HTTP requests
to the retriever), this manager:
1. Generates responses for ALL active samples concurrently via the async LLM server
2. Collects ALL search queries into a single batch
3. Sends ONE HTTP POST to the retrieval server
4. Distributes results back to the appropriate samples

This reduces retriever overhead from O(batch_size * n * turns) calls to O(turns) calls.
"""

import asyncio
import importlib
import re
import time
from uuid import uuid4

import numpy as np
import requests
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    _get_rollout_and_model_config,
)
from verl.utils.ray_utils import auto_await


def compute_position_id_with_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask


class BatchedSearchAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that uses batched search for multi-turn Search-R1 training."""

    def __init__(self, config, worker_group=None, rollout_resource_pool=None, reward_loop_worker_handles=None):
        super().__init__(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)

        retriever_cfg = config.get("retriever", {})
        self.search_url = retriever_cfg.get("url", "http://localhost:8000/retrieve")
        self.topk = retriever_cfg.get("topk", 3)

        mt_cfg = config.actor_rollout_ref.rollout.get("multi_turn", {})
        self.max_turns = mt_cfg.get("max_assistant_turns", 2)
        self.max_obs_length = mt_cfg.get("max_tool_response_length", 500)

        reward_cfg = config.reward.custom_reward_function
        if reward_cfg.path:
            spec = importlib.util.spec_from_file_location("reward_fn", reward_cfg.path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.reward_fn = getattr(mod, reward_cfg.name)
        else:
            self.reward_fn = None

    async def _init_agent_loop_workers(self):
        """Override: we don't use per-sample agent loop workers.
        Instead, set up the server manager and tokenizer on this manager directly."""
        from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
        from verl.utils.config import omega_conf_to_dataclass

        _, model_config = _get_rollout_and_model_config(self.config)
        self.model_config = omega_conf_to_dataclass(model_config)
        self.tokenizer = self.model_config.tokenizer

        servers = list(zip(self.server_addresses, self.server_handles, strict=True))
        self.server_manager = AsyncLLMServerManager(
            self.config,
            servers,
            load_balancer_handle=self.global_load_balancer,
        )

        # Fake workers list so len(self.agent_loop_workers) doesn't fail
        self.agent_loop_workers = []

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        t_start = time.time()
        config = self.rollout_config
        batch_size = len(prompts)

        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "max_tokens": config.response_length,
            "stop": ["</search>", "</answer>"],
        }

        if prompts.meta_info.get("validate", False):
            sampling_params["temperature"] = config.val_kwargs.temperature
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k

        pad_id = self.tokenizer.pad_token_id

        # Tokenize from raw_prompt (list of message dicts) since upstream verl
        # passes non_tensor_batch only (batch is None after _get_gen_batch)
        original_prompt_ids = []
        for i in range(batch_size):
            raw_prompt = prompts.non_tensor_batch["raw_prompt"][i]
            text = self.tokenizer.apply_chat_template(
                raw_prompt, tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            original_prompt_ids.append(ids)

        # State tracking per sample
        current_prompt_ids = [list(ids) for ids in original_prompt_ids]
        all_response_ids = [[] for _ in range(batch_size)]
        all_response_mask = [[] for _ in range(batch_size)]
        active_mask = [True] * batch_size
        num_turns = [0] * batch_size
        request_ids = [uuid4().hex for _ in range(batch_size)]

        t_gen_total = 0.0
        t_search_total = 0.0

        for turn in range(self.max_turns + 1):
            is_final_turn = turn == self.max_turns
            active_indices = [i for i in range(batch_size) if active_mask[i]]
            if not active_indices:
                break

            # Generate responses for all active samples concurrently
            t_gen_start = time.time()
            gen_params = dict(sampling_params)
            if is_final_turn:
                gen_params["stop"] = ["</answer>"]

            tasks = []
            for i in active_indices:
                tasks.append(
                    self.server_manager.generate(
                        request_ids[i],
                        prompt_ids=current_prompt_ids[i],
                        sampling_params=gen_params,
                    )
                )
            outputs = await asyncio.gather(*tasks)
            t_gen_total += time.time() - t_gen_start

            # Postprocess: decode, truncate at stop tags, re-tokenize
            responses_str = []
            responses_token_ids = []
            for idx, output in enumerate(outputs):
                token_ids = output.token_ids
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                if "</search>" in text:
                    text = text.split("</search>")[0] + "</search>"
                elif "</answer>" in text:
                    text = text.split("</answer>")[0] + "</answer>"
                responses_str.append(text)
                re_tokenized = self.tokenizer.encode(text, add_special_tokens=False)
                responses_token_ids.append(re_tokenized)

            # Parse actions
            actions = []
            contents = []
            for text in responses_str:
                match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
                if match:
                    actions.append(match.group(1))
                    contents.append(match.group(2).strip())
                else:
                    actions.append(None)
                    contents.append("")

            # Batch search
            t_search_start = time.time()
            search_queries = [c for a, c in zip(actions, contents) if a == "search" and not is_final_turn]
            search_results = []
            if search_queries:
                search_results = self._batch_search(search_queries)
            t_search_total += time.time() - t_search_start

            # Update per-sample state
            search_idx = 0
            for local_idx, i in enumerate(active_indices):
                action = actions[local_idx]
                resp_ids = responses_token_ids[local_idx]

                all_response_ids[i].extend(resp_ids)
                all_response_mask[i].extend([1] * len(resp_ids))
                num_turns[i] += 1

                if action == "answer" or is_final_turn:
                    active_mask[i] = False
                elif action == "search":
                    obs_text = f"\n\n<information>{search_results[search_idx].strip()}</information>\n\n"
                    search_idx += 1
                    obs_ids = self.tokenizer.encode(obs_text, add_special_tokens=False)
                    if len(obs_ids) > self.max_obs_length:
                        obs_ids = obs_ids[: self.max_obs_length]

                    all_response_ids[i].extend(obs_ids)
                    all_response_mask[i].extend([0] * len(obs_ids))

                    current_prompt_ids[i] = original_prompt_ids[i] + all_response_ids[i]
                else:
                    obs_text = (
                        "\nMy previous action is invalid. "
                        "If I want to search, I should put the query between <search> and </search>. "
                        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                    obs_ids = self.tokenizer.encode(obs_text, add_special_tokens=False)
                    all_response_ids[i].extend(obs_ids)
                    all_response_mask[i].extend([0] * len(obs_ids))

                    current_prompt_ids[i] = original_prompt_ids[i] + all_response_ids[i]

            if search_queries:
                assert search_idx == len(search_results)

        # Assemble output DataProto
        prompt_length = config.prompt_length
        response_length = config.response_length

        # Truncate responses that exceed budget
        for i in range(batch_size):
            if len(all_response_ids[i]) > response_length:
                all_response_ids[i] = all_response_ids[i][:response_length]
                all_response_mask[i] = all_response_mask[i][:response_length]

        # Left-pad prompts, right-pad responses
        prompt_tensors = []
        response_tensors = []
        response_mask_tensors = []

        for i in range(batch_size):
            p_ids = original_prompt_ids[i]
            if len(p_ids) > prompt_length:
                p_ids = p_ids[-prompt_length:]
            pad_len = prompt_length - len(p_ids)
            prompt_tensors.append([pad_id] * pad_len + p_ids)

            r_ids = all_response_ids[i]
            r_mask = all_response_mask[i]
            pad_len = response_length - len(r_ids)
            response_tensors.append(r_ids + [pad_id] * pad_len)
            response_mask_tensors.append(r_mask + [0] * pad_len)

        prompts_t = torch.tensor(prompt_tensors, dtype=torch.long)
        responses_t = torch.tensor(response_tensors, dtype=torch.long)
        response_mask_t = torch.tensor(response_mask_tensors, dtype=torch.long)

        input_ids_t = torch.cat([prompts_t, responses_t], dim=1)
        prompt_attn = (prompts_t != pad_id).long()
        response_attn = (responses_t != pad_id).long()
        attention_mask_t = torch.cat([prompt_attn, response_attn], dim=1)
        response_mask_t = response_mask_t * response_attn
        position_ids_t = compute_position_id_with_mask(attention_mask_t)

        # Compute rm_scores using the custom reward function
        rm_scores = torch.zeros_like(response_mask_t, dtype=torch.float32)
        if self.reward_fn is not None and "reward_model" in prompts.non_tensor_batch:
            for i in range(batch_size):
                resp_text = self.tokenizer.decode(all_response_ids[i], skip_special_tokens=False)
                reward_model_info = prompts.non_tensor_batch["reward_model"][i]
                ground_truth = reward_model_info.get("ground_truth", reward_model_info)
                data_source = prompts.non_tensor_batch.get("data_source", np.array([""] * batch_size))[i]
                score = self.reward_fn(data_source, resp_text, ground_truth)
                # Place score at last valid response token
                resp_len = int(response_attn[i].sum().item())
                if resp_len > 0:
                    rm_scores[i, resp_len - 1] = score

        batch_dict = TensorDict(
            {
                "prompts": prompts_t,
                "responses": responses_t,
                "response_mask": response_mask_t,
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "position_ids": position_ids_t,
                "rm_scores": rm_scores,
            },
            batch_size=batch_size,
        )

        non_tensor_batch = {
            "__num_turns__": np.array(num_turns, dtype=np.int32),
            "multi_modal_inputs": np.array([{} for _ in range(batch_size)], dtype=object),
        }
        if prompts.non_tensor_batch:
            for k, v in prompts.non_tensor_batch.items():
                if k not in non_tensor_batch:
                    non_tensor_batch[k] = v

        t_total = time.time() - t_start
        timing = {
            "agent_loop/generate_sequences/mean": t_gen_total / max(batch_size, 1),
            "agent_loop/generate_sequences/min": t_gen_total / max(batch_size, 1),
            "agent_loop/generate_sequences/max": t_gen_total / max(batch_size, 1),
            "agent_loop/tool_calls/mean": t_search_total / max(batch_size, 1),
            "agent_loop/tool_calls/min": t_search_total / max(batch_size, 1),
            "agent_loop/tool_calls/max": t_search_total / max(batch_size, 1),
            "agent_loop/num_preempted/mean": 0.0,
            "agent_loop/num_preempted/min": 0,
            "agent_loop/num_preempted/max": 0,
            "agent_loop/slowest/generate_sequences": t_gen_total,
            "agent_loop/slowest/tool_calls": t_search_total,
            "agent_loop/slowest/prompt_length": prompt_length,
            "agent_loop/slowest/response_length": response_length,
            "agent_loop/slowest/num_preempted": 0,
        }

        return DataProto(
            batch=batch_dict,
            non_tensor_batch=non_tensor_batch,
            meta_info={"timing": timing},
        )

    def _batch_search(self, queries: list[str], chunk_size: int = 256) -> list[str]:
        if not queries:
            return []
        all_formatted = []
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start : start + chunk_size]
            payload = {"queries": chunk, "topk": self.topk, "return_scores": True}
            resp = requests.post(self.search_url, json=payload, timeout=600)
            results = resp.json()["result"]
            for result in results:
                text = ""
                for idx, doc in enumerate(result):
                    content = doc["document"]["contents"]
                    title = content.split("\n")[0]
                    body = "\n".join(content.split("\n")[1:])
                    text += f"Doc {idx+1}(Title: {title}) {body}\n"
                all_formatted.append(text)
        return all_formatted
