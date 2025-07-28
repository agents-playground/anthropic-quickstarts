"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import logging
import os
import pickle
import sqlite3
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from typing import cast

import httpx
import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from streamlit.delta_generator import DeltaGenerator

from computer_use_demo.loop import (
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.groups import COMPUTER_USE_20250429

logger = logging.Logger("streamlit.py")
logging.basicConfig(level=logging.INFO)


@dataclass(kw_only=True, frozen=True)
class ModelConfig:
    max_output_tokens: int
    default_output_tokens: int
    has_thinking: bool = False

CLAUDE_4 = ModelConfig(
    max_output_tokens=128_000,
    default_output_tokens=1024 * 16,
    has_thinking=True,
)

INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"

IN_SAMPLING_LOOP = False


class Sender:
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


DB_FILE = os.path.join(tempfile.gettempdir(), "state.sqlite3")

MAX_HISTORY_SIZE = 3

def setup_state():
    st.session_state.messages = []
    st.session_state.api_key = os.getenv("ANTHROPIC_API_KEY")
    st.session_state.provider = APIProvider.ANTHROPIC
    model_conf = CLAUDE_4
    st.session_state.model = "claude-sonnet-4-20250514"
    st.session_state.output_tokens = model_conf.default_output_tokens
    st.session_state.max_output_tokens = model_conf.max_output_tokens
    st.session_state.responses = {}
    st.session_state.tools = {}
    st.session_state.custom_system_prompt = ""
    st.session_state.token_efficient_tools_beta = False

    with sqlite3.connect(DB_FILE) as connection:
        cursor = connection.cursor()
        # create tables if not exist
        with open("computer_use_demo/tables.sql", "r") as f:
            ddl = f.read()
            cursor.executescript(ddl)
            connection.commit()
        # read msgs
        cursor.execute("SELECT state FROM session_state WHERE id = 1")
        row = cursor.fetchone()
        if row:
            for k, v in pickle.loads(row[0]).items():
                st.session_state[k] = v


def reset_db():
    with sqlite3.connect(DB_FILE) as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM session_state")
        connection.commit()


async def main():
    """Render loop for streamlit"""
    setup_state()
    st.title("Claude Computer Use")

    with st.sidebar:
        st.text_area(
            "Custom System Prompt Suffix",
            key="custom_system_prompt",
            help="Additional instructions to append to the system prompt",
            on_change=lambda: persist_state(),
        )
        st.checkbox(
            "Enable token-efficient tools beta",
            key="token_efficient_tools_beta",
            on_change=lambda: persist_state(),
        )
        if st.button("Reset conversation history", type="primary"):
            with st.spinner("Resetting..."):
                st.session_state.clear()
                reset_db()
                setup_state()

    chat_tab, requests_tab = st.tabs(["Chat", "HTTP Exchange Logs"])
    new_message = st.chat_input("Type a message to Claude...")

    with chat_tab:
        # render past chats
        for message in st.session_state.messages[-10:]:
            if isinstance(message["content"], str):
                _render_message(message["role"], message["content"])
            elif isinstance(message["content"], list):
                for block in message["content"]:
                    # the tool result we send back to the Anthropic API isn't sufficient to render all details,
                    # so we store the tool use responses
                    if isinstance(block, dict) and block["type"] == "tool_result":
                        _render_message(
                            Sender.TOOL, st.session_state.tools[block["tool_use_id"]]
                        )
                    else:
                        _render_message(
                            message["role"],
                            cast(BetaContentBlockParam | ToolResult, block),
                        )

        # render past http exchanges
        for identity, (request, response) in list(st.session_state.responses.items())[-3:]:
            _render_api_response(request, response, identity, requests_tab)

        # render past chats
        if new_message:
            st.session_state.messages.append(
                {
                    "role": Sender.USER,
                    "content": [
                        *maybe_add_interruption_blocks(),
                        BetaTextBlockParam(type="text", text=new_message),
                    ],
                }
            )
            _render_message(Sender.USER, new_message)
            slice_chat_history()

        if not st.session_state.messages or st.session_state.messages[-1]["role"] != Sender.USER:
            # we don't have a user message to respond to, exit early
            return

        with track_sampling_loop():
            persist_state()
            # run the agent sampling loop with the newest message
            st.session_state.messages = await sampling_loop(
                system_prompt_suffix=st.session_state.custom_system_prompt,
                model=st.session_state.model,
                messages=st.session_state.messages,
                output_callback=partial(_render_message, Sender.BOT),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=st.session_state.tools
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=requests_tab,
                    response_state=st.session_state.responses,
                ),
                api_key=st.session_state.api_key,
                max_tokens=st.session_state.output_tokens,
                token_efficient_tools_beta=st.session_state.token_efficient_tools_beta,
                persist_state=persist_state
            )
            persist_state()


def persist_state():
    pickled_state = pickle.dumps(st.session_state.to_dict())
    with sqlite3.connect(DB_FILE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO session_state (id, state) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET state = ?, timestamp = CURRENT_TIMESTAMP", (pickled_state, pickled_state))
        connection.commit()


def maybe_add_interruption_blocks():
    if not IN_SAMPLING_LOOP:
        return []
    # If this function is called while we're in the sampling loop, we can assume that the previous sampling loop was interrupted
    # and we should annotate the conversation with additional context for the model and heal any incomplete tool use calls
    result = []
    last_message = st.session_state.messages[-1]
    previous_tool_use_ids = [
        block["id"] for block in last_message["content"] if block["type"] == "tool_use"
    ]
    for tool_use_id in previous_tool_use_ids:
        st.session_state.tools[tool_use_id] = ToolResult(error=INTERRUPT_TOOL_ERROR)
        result.append(
            BetaToolResultBlockParam(
                tool_use_id=tool_use_id,
                type="tool_result",
                content=INTERRUPT_TOOL_ERROR,
                is_error=True,
            )
        )
    result.append(BetaTextBlockParam(type="text", text=INTERRUPT_TEXT))
    return result


@contextmanager
def track_sampling_loop():
    global IN_SAMPLING_LOOP
    IN_SAMPLING_LOOP = True
    yield
    IN_SAMPLING_LOOP = False


def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    """
    Handle an API response by storing it to state and rendering it.
    """
    slice_chat_history()
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _render_error(error)
    _render_api_response(request, response, response_id, tab)


def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state and rendering it."""
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output)


def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    """Render an API response to a streamlit tab"""
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception):
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_message(
    sender: str,
    message: str | BetaContentBlockParam | ToolResult,
):
    """Convert input from the user or output from the agent to a streamlit message."""
    # streamlit's hot reloading breaks isinstance checks, so we need to check for class names
    is_tool_result = not isinstance(message, str | dict)
    with st.chat_message(sender):
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                if message.__class__.__name__ == "CLIResult":
                    st.code(message.output)
                else:
                    st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image:
                st.image(base64.b64decode(message.base64_image))
        elif isinstance(message, dict):
            if message["type"] == "text":
                st.write(message["text"])
            elif message["type"] == "tool_use":
                st.code(f'Tool Use: {message["name"]}\nInput: {message["input"]}')
            else:
                # only expected return types are text and tool_use
                raise Exception(f'Unexpected response type {message["type"]}')
        else:
            st.markdown(message)


def slice_chat_history():
    pass
    # if len(st.session_state.messages) > MAX_HISTORY_SIZE:
    #     st.session_state.messages = st.session_state.messages[-MAX_HISTORY_SIZE:]


if __name__ == "__main__":
    asyncio.run(main())
