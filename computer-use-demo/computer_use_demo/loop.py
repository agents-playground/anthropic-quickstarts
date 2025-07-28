"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""
import logging
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import httpx
from anthropic import (
    Anthropic,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import (
    ToolCollection,
    ToolResult,
)
from .tools.groups import ToolGroup, COMPUTER_USE_20250429

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"

logger = logging.Logger("loop.py")
logging.basicConfig(level=logging.INFO)


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your str_replace_based_edit_tool.
</IMPORTANT>"""


async def sampling_loop(
    *,
    model: str,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    max_tokens: int = 4096,
    token_efficient_tools_beta: bool = False,
    persist_state: Callable = None
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_group = COMPUTER_USE_20250429
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        logger.warning(f"{datetime.now()} loop.py iteration ...")
        betas = [tool_group.beta_flag] if tool_group.beta_flag else []
        if token_efficient_tools_beta:
            betas.append("token-efficient-tools-2025-02-19")

        client = Anthropic(api_key=api_key, max_retries=50)
        betas.append(PROMPT_CACHING_BETA_FLAG)
        system["cache_control"] = {"type": "ephemeral"}  # type: ignore

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            logger.warning(f"{datetime.now()} Sending request to {model} ...")
            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=tool_collection.to_params(),
                betas=betas,
            )

            del messages[-1]["content"][-1]["cache_control"]
            logger.warning(f"{datetime.now()} Received {model} response.")
        except (APIStatusError, APIResponseValidationError) as e:
            api_response_callback(e.request, e.response, e)
            output_callback("Exception raised. Execution won't continue without new user input.")
            logger.warning(f"{datetime.now()} Exception raised.", e)
            return messages
        except APIError as e:
            api_response_callback(e.request, e.body, e)
            output_callback("Exception raised. Execution won't continue without new user input.")
            logger.warning(f"{datetime.now()} Exception raised.", e)
            return messages

        api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )

        response = raw_response.parse()
        response_params = _response_to_params(response)
        messages.append({"role": "assistant", "content": response_params})

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                tool_name = content_block["name"]
                tool_input = cast(dict[str, Any], content_block["input"])
                logger.warning(f"{datetime.now()} Running {tool_name} tool. Input: {tool_input} ...")
                result = await tool_collection.run(
                    name=tool_name,
                    tool_input=tool_input,
                )
                logger.warning(f"{datetime.now()} {tool_name} tool done.")
                tool_result_content.append(_make_api_tool_result(result, content_block["id"]))
                tool_output_callback(result, content_block["id"])

        if not tool_result_content:
            output_callback("No tool use requested. Prev response is final. Execution terminated.")
            logger.warning(f"{datetime.now()} No tool use requested.")
            return messages

        messages.append({"content": tool_result_content, "role": "user"})
        persist_state()


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    res: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
        else:
            # Handle tool use blocks normally
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
