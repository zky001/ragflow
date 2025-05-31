# Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP retrieval component.

This component uses a running MCP server to fetch chunks from
remote datasets via the ``ragflow_retrieval`` tool.
"""

from __future__ import annotations

import asyncio
from abc import ABC
from typing import Any, Dict, List

import pandas as pd

try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
except Exception:  # pragma: no cover - MCP client may not be installed
    ClientSession = None  # type: ignore
    sse_client = None  # type: ignore

from agent.component.base import ComponentBase, ComponentParamBase


class MCPRetrievalParam(ComponentParamBase):
    """Parameters for :class:`MCPRetrieval`."""

    def __init__(self) -> None:
        super().__init__()
        self.mcp_url: str = "http://localhost:9382/sse"
        self.dataset_ids: List[str] = []
        self.document_ids: List[str] = []

    def check(self) -> None:  # pragma: no cover - simple check
        if not isinstance(self.mcp_url, str):
            raise ValueError("mcp_url should be a string")


class MCPRetrieval(ComponentBase, ABC):
    """Call ``ragflow_retrieval`` on an MCP server."""

    component_name = "MCPRetrieval"

    async def _async_call_tool(self, question: str) -> Any:
        if ClientSession is None or sse_client is None:
            raise RuntimeError("MCP client library is not available")

        async with sse_client(self._param.mcp_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                response = await session.call_tool(
                    name="ragflow_retrieval",
                    arguments={
                        "dataset_ids": self._param.dataset_ids,
                        "document_ids": self._param.document_ids,
                        "question": question,
                    },
                )
                return response

    def _run(self, history, **kwargs):  # noqa: D401 - interface defined in base
        ans = self.get_input()
        question = str(ans["content"][0]) if "content" in ans else ""
        if not question:
            return MCPRetrieval.be_output("")

        try:
            res = asyncio.run(self._async_call_tool(question))
        except Exception as e:  # pragma: no cover - network failures
            return MCPRetrieval.be_output(f"Error: {e}")

        try:
            outputs = res.outputs  # type: ignore[attr-defined]
        except AttributeError:
            try:
                res_dict: Dict[str, Any] = res.model_dump()  # type: ignore[attr-defined]
                outputs = res_dict.get("outputs", [])
            except Exception:
                outputs = []

        contents = []
        for item in outputs:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text is None:
                text = str(item)
            contents.append(text)

        if not contents:
            return MCPRetrieval.be_output("")
        return pd.DataFrame({"content": contents})
