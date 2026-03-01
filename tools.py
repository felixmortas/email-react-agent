from typing import Annotated

from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from context import Context
from browser_helpers import _read_page_html, _click_element, _fill_text_field, _update_progress_step

@tool
async def read_page_html(runtime: ToolRuntime[Context]) -> str:
    """
    Returns compact page elements: <tag>text</tag> or <tag id="x" class="y" type="z">
    """
    html_content = await _read_page_html(runtime)

    return f"HTML Content:\n{html_content}"



@tool
async def click_element(
    runtime: ToolRuntime[Context],
    text: str = "",
    tag: str = "",
    id_name: str = "",
    class_name: str = "",
    type_name: str = ""
) -> str:
    """
    Clicks a DOM element identified by a unique combination of attributes, then returns the updated page elements.

    Provide at least tag and another attribute to uniquely identify the target element.

    Exemple: text, tag OR tag, class_name, type_name

    Args:
        text: Visible text content of the element (e.g. link text); [REQUIRED if possible]
        tag: HTML tag name of the element (e.g. "button", "a", "input"); [REQUIRED]
        id_name: The element's `id` attribute; [REQUIRED if possible]
        class_name: One or more CSS classes on the element (space-separated); [REQUIRED if possible]
        type_name: The element's `type` attribute (e.g. "submit", "checkbox"). [REQUIRED if possible]

    Returns:
        A compact representation of the page elements after the click.
    """

    click_response = await _click_element(runtime, text, tag, id_name, class_name, type_name)
    await runtime.context["page"].wait_for_load_state("networkidle")
    html_content = await _read_page_html(runtime)

    return f"{click_response}\n\nHTML Content:\n{html_content}"
    
@tool
async def fill_text_field(
    runtime: ToolRuntime[Context],
    identifier: str,
    element_tag: str,
    element_id: str = "",
    element_class: str = "",
    element_type: str = ""
) -> str:
    """
    Fills field using ONE UNIQUE selector.
    The selector is choosed using a unique combination of: tag and/or id and/or class and/or type.
    The 'identifier' parameter determines which credential to use.
    Read the HTML page again if the filling action fails.
        
    Args:
        identifier: 'EMAIL' | 'PASSWORD' | 'NEW_EMAIL' (REQUIRED)
        element_tag: HTML tag from read_page_html (e.g., 'input') (REQUIRED)
        element_id: HTML id from read_page_html (e.g., 'email', 'passwd')
        element_class: HTML class from read_page_html (e.g., 'account_input')
        element_type: HTML type from read_page_html (e.g., 'email', 'password')

    """
    fill_response = await _fill_text_field(runtime, identifier, element_tag, element_id, element_class, element_type)

    if fill_response.startswith('❌'):
        html_content = await _read_page_html(runtime)
        return f"{fill_response}\n\nHTML Content:\n{html_content}"
    return fill_response



@tool
async def close_popup(runtime: ToolRuntime[Context]) -> str:
    """
    Tries to close an active popup or modal using common selectors.
    """
    page = runtime.context["page"]
    selectors = [
        "button[aria-label='Close']", 
        ".close", 
        "button:text('Close')",
        "button:text('OK')",
        "[role='dialog'] button:first-child"
    ]
    
    for selector in selectors:
        if await page.is_visible(selector):
            await page.click(selector)
            return f"Popup closed with selector: {selector}"
    
    return "No popup detected with common selectors"

@tool
async def update_progress_state(step: str, runtime: ToolRuntime[Context]) -> str:
    """
    Signals that the step in progress has been completed.

    Args: 
        step: Step of the CURRENT PROGRESS STATE.
    """
    update_response = await _update_progress_step(step=step)
    html_content = await _read_page_html(runtime)

    return f"{update_response}\n\nHTML Content: {html_content}"



tools = [
    read_page_html,
    click_element,
    fill_text_field,
    # close_popup,
    update_progress_state,
]