"""
Custom ToolParser for Search-R1's <search>query</search> format.
Register as 'search_r1' so verl's agent loop can parse search tags.

Works with stop=["</search>"] — generation halts at </search>,
so the response may contain "<search>query" without the closing tag.
"""

import json
import logging
import os
import regex

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("search_r1")
class SearchR1ToolParser(ToolParser):
    """Parses <search>query</search> tags from Search-R1 style generations.
    Handles both complete tags and truncated ones (when stop=["</search>"])."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Match complete <search>...</search> or <search>... at end of string
        self.search_complete = regex.compile(r"<search>(.*?)</search>", regex.DOTALL)
        self.search_open = regex.compile(r"<search>(.*?)$", regex.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids, tools=None):
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        # Try complete tags first
        matches = self.search_complete.findall(text)
        if not matches:
            # Try open tag (generation stopped at </search>)
            matches = self.search_open.findall(text)

        if not matches:
            return text, []

        # Only take the FIRST search query to prevent search-spamming
        first_query = next((m.strip() for m in matches if m.strip()), None)
        if not first_query:
            return text, []

        function_calls = [
            FunctionCall(
                name="search",
                arguments=json.dumps({"query_list": [first_query]}, ensure_ascii=False),
            )
        ]

        content = self.search_complete.sub("", text)
        content = self.search_open.sub("", content)
        return content, function_calls
