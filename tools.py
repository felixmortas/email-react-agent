import json
import os
from langchain.tools import tool, ToolRuntime
from context import Context

@tool
async def read_page_html(runtime: ToolRuntime[Context]) -> str:
    """
    Returns compact page elements: <tag>text</tag> or <tag id="x" class="y">
    Minimizes tokens while keeping essential info for LLM decisions.
    """
    page = runtime.context["page"]
    elements = await page.evaluate("""() => {
        const sel = 'button, a, input, [role="button"], [role="alert"]';
        return Array.from(document.querySelectorAll(sel))
            .filter(el => el.offsetWidth > 0 && el.offsetHeight > 0)
            .map(el => {
                const tag = el.tagName.toLowerCase();
                const text = (el.innerText || '').trim();
                const id = el.id ? ` id="${el.id}"` : '';
                const cls = el.className ? ` class="${el.className}"` : '';
                const type = el.type ? ` type="${el.type}"` : '';
;
                
                if (text) {
                    return `<${tag}${type}>${text}</${tag}>`;
                } else {
                    return `<${tag}${id}${cls}${type}>`;
                }
            })
    }""")
    return "\n".join(elements)

@tool
async def click_element(
    runtime: ToolRuntime[Context],
    text: str = "",
    tag: str = "",
    id: str = "",
    class_name: str = "",
    type_name: str = ""
) -> str:
    """
    Clicks element using exact match strategies:
    - Mode 1: text + tag
    - Mode 2: tag + id/class
    Clicks first visible element only. No partial/regex matching.
    """
    page = runtime.context["page"]
    
    # ─────────────────────────────────────────────────────────────
    # MODE 1: text + tag → <tag>text</tag>
    # ─────────────────────────────────────────────────────────────
    if text and tag:
        selector = f"{tag}:has-text('{text}')"
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0 and await locator.is_visible():
                await locator.scroll_into_view_if_needed()
                await locator.click(timeout=5000)
                return f"✅ Clicked: <{tag}>{text}</{tag}>"
        except Exception:
            pass
        return f"❌ Not found: <{tag}>{text}</{tag}>"
    
    # ─────────────────────────────────────────────────────────────
    # MODE 2: tag + id/class → <tag id='x' class='y'>
    # ─────────────────────────────────────────────────────────────
    if tag and (id or class_name or type):
        selector = tag
        if id:
            selector += f"#{id}"
        if class_name:
            # Take only first class to keep selector simple
            selector += f".{class_name.split(' ')[0]}"
        if type_name:
            selector += f"[type='{type_name}']"
        
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0 and await locator.is_visible():
                await locator.scroll_into_view_if_needed()
                await locator.click(timeout=5000)
                
                # Build response matching read_page_html format
                attrs = ""
                if id:
                    attrs += f" id=\"{id}\""
                if class_name:
                    attrs += f" class=\"{class_name}\""
                return f"✅ Clicked: <{tag}{attrs}>"
        except Exception:
            pass
        
        # Build error response
        attrs = ""
        if id:
            attrs += f" id=\"{id}\""
        if class_name:
            attrs += f" class=\"{class_name}\""
        return f"❌ Not found: <{tag}{attrs}>"
    
    # ─────────────────────────────────────────────────────────────
    # INVALID PARAMETERS
    # ─────────────────────────────────────────────────────────────
    return "❌ Invalid params: provide (text + tag) OR (tag + id/class)"

@tool
async def fill_text_field(
    runtime: ToolRuntime[Context],
    tag: str,
    identifier: str,
    element_id: str = "",
    element_class: str = "",
    element_type: str = ""
) -> str:
    """
    ⚠️ CREDENTIALS ARE AUTO-FILLED FROM os.environ - DO NOT ASK FOR THEM!
    
    Fills field using ONE selector: tag + id + class + type.
    The 'identifier' parameter determines which credential to use:
    
    | identifier  | Usage                    |
    |-------------|--------------------------|
    | 'EMAIL'     | Current email/username   |
    | 'PASSWORD'  | Password                 |
    | 'NEW_EMAIL' | New email to set         |
    
    Args:
        tag: HTML tag from read_page_html (e.g., 'input')
        identifier: 'EMAIL' | 'PASSWORD' | 'NEW_EMAIL' (REQUIRED)
        element_id: HTML id from read_page_html (e.g., 'email', 'passwd')
        element_class: HTML class from read_page_html (e.g., 'account_input')
        element_type: HTML type from read_page_html (e.g., 'email', 'password')
    
    ✅ EXAMPLES:
        fill_text_field(tag='input', identifier='EMAIL', element_id='email', element_class='account_input', element_type='email')
        fill_text_field(tag='input', identifier='PASSWORD', element_id='passwd', element_class='account_input', element_type='password')
        fill_text_field(tag='input', identifier='NEW_EMAIL', element_id='new-email', element_class='form-input', element_type='email')
    
    ❌ NEVER:
        - Ask user for credentials
        - Pass actual password/email values
        - Use field_text or value parameter
    """
    page = runtime.context["page"]
    
    value = os.environ.get(identifier, "")
    
    if not value:
        return f"❌ os.environ['{identifier}'] is empty"
    
    # Build SINGLE selector from tag + id + class + type
    if element_id:
        if element_class:
            selector = f"{tag}#{element_id}.{element_class.split(' ')[0]}"
        else: 
            selector = f"{tag}#{element_id}"
    elif element_class:
        selector = f"{tag}.{element_class.split(' ')[0]}"
    elif element_type:
        selector = f"{tag}[type='{element_type}']"
    else:
        selector = tag  # fallback minimal

    if element_type and (element_id or element_class):
        selector += f"[type='{element_type}']"
    
    # Fill with single selector
    try:
        locator = page.locator(selector).first
        if await locator.count() > 0 and await locator.is_visible():
            await locator.fill(value, timeout=5000)
            return f"✅ Filled <{tag} id='{element_id or '*'}' class='{element_class or '*'}' type='{element_type or '*'}'>"
        return f"❌ Element not visible: {selector}"
    except Exception as e:
        return f"❌ Failed: {selector} - {str(e)[:50]}"
    
@tool
async def submit_form(runtime: ToolRuntime[Context]) -> str:
    """
    Submits form using most reliable method.
    Priority: Enter key > Submit button > First button.
    """
    page = runtime.context["page"]
    
    try:
        # Method 1: Enter on focused input
        await page.press("input:focus", "Enter")
        return "✅ Submitted: Enter key"
    except:
        pass
    
    try:
        # Method 2: Submit button
        submit_btn = page.locator("button[type='submit'], input[type='submit']")
        if await submit_btn.count() > 0:
            await submit_btn.first.click(timeout=3000)
            return "✅ Submitted: Submit button"
    except:
        pass
    
    try:
        # Method 3: Any button with submit-like text
        submit_keywords = ["submit", "valider", "ok", "continuer", "suivant"]
        for keyword in submit_keywords:
            btn = page.locator(f"button:has-text('{keyword}' i)").first
            if await btn.count() > 0:
                await btn.click(timeout=3000)
                return f"✅ Submitted: Button '{keyword}'"
    except:
        pass
    
    return "❌ Submit failed"

@tool
async def navigate_to_url(runtime: ToolRuntime[Context], url: str) -> str:
    """
    Directs the browser to a specific URL. Ensure the URL is complete (starting with http/https).
    """
    page = runtime.context["page"]
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    return f"Navigated to {url}"

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
async def get_current_url(runtime: ToolRuntime[Context]) -> str:
    """
    Returns the current URL and the page title.
    """
    page = runtime.context["page"]
    title = await page.title()
    return json.dumps({"url": page.url, "title": title})

@tool
async def mark_task_complete(reason: str) -> str:
    """
    Signals that the objective has been reached and provides the final summary.
    """
    return f"GOAL REACHED: {reason}"

tools = [
    read_page_html,
    click_element,
    fill_text_field,
    submit_form,
    navigate_to_url,
    close_popup,
    get_current_url,
    mark_task_complete,
]