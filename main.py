"""
Main Entry Point for ReAct Agent Email Changer
The LLM generates all the logic - minimal code, maximum flexibility
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
from playwright.async_api import async_playwright
from contextlib import asynccontextmanager
import asyncio
from langchain.messages import HumanMessage

from langfuse_engine import langfuse_handler
from agent import create_email_agent
from browser_helpers import extract_semantic_html
from context import Context

# ==================== PLAYWRIGHT CONTEXT MANAGER ====================


@asynccontextmanager
async def playwright_session(headless: bool = True):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, slow_mo=500)
        page = await browser.new_page()
        try:
            yield page
        finally:
            await page.close()
            await browser.close()


# ==================== WORKFLOW ORCHESTRATOR ====================

async def run_email_change_workflow(
    website_url: str,
    website_name: str = "Website",
    agent = None,
    headless: bool = True,
):
    """
    Complete workflow: Login → Navigate to Settings → Change Email
    
    Args:
        website_url: Starting URL
        website_name: Name of the website (for context)
    """
    print("\n" + "="*80)
    print(f"🚀 EMAIL CHANGE WORKFLOW FOR: {website_name}")
    print("="*80)
        
    async with playwright_session(headless=headless) as page:
        await page.goto(url=website_url, wait_until="domcontentloaded")
        html_content = await extract_semantic_html(page)

        inputs = {
            "messages": [
                HumanMessage(f"HTML content of the starting page for website {website_name}\n\n{html_content}")
            ],
            "current_url": website_url,
            "last_step_url": website_url,
        }


        await agent.ainvoke(inputs, context=Context(page=page, website_url=website_url), 
            config={
                "callbacks": [langfuse_handler], 
                "metadata": {"langfuse_tags": [website_name]}
            }
        )
        


# ==================== MAIN ====================

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="ReAct Agent for automated email changing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --url https://www.agrosemens.com --website Agrosemens --model mistral-large-latest --no-headless
  python main.py --url https://www.agrosemens.com --website Agrosemens --model mistral-large-latest
        """
    )
    
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Website URL to start from"
    )
    
    parser.add_argument(
        "--website",
        type=str,
        required=True,
        help="Website name for context"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistral-large-latest",
        help="Model name (e.g. mistral-large-latest, gemini-3-flash-preview)"
    )

    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run browser in headless mode"
    )

    parser.add_argument(
        "--eco",
        action="store_true",
        default=False,
        help="Enable eco mode"
    )

    args = parser.parse_args()
    
    # Get credentials from args or environment
    username = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    new_email = os.getenv("NEW_EMAIL")
    
    # Validate
    if not username or not password:
        print("❌ Error: Username and password required (via args or .env)")
        print("   Set EMAIL and PASSWORD in .env")
        return
    
    if not new_email:
        print("❌ Error: New email required (via args or .env)")
        print("   Set NEW_EMAIL in .env or use --new-email")
        return
    
    # Run the workflow
    agent = create_email_agent(model_name=args.model, eco=args.eco)

    asyncio.run(run_email_change_workflow(
        website_url=args.url,
        website_name=args.website,
        agent=agent,
        headless=args.headless,
    ))
    

if __name__ == "__main__":
    main()