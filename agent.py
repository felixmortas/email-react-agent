from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, ContextEditingMiddleware, ClearToolUsesEdit
from langchain_mistralai import ChatMistralAI
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from tools import tools
from system_prompt import dynamic_system_prompt, system_prompt
from state import State
from context import Context

import os
from dotenv import load_dotenv
load_dotenv()

# Middleware to keep only the last 2 messages
@before_model
def trim_messages(state: State, runtime: ToolRuntime[Context]):
    """
    Keep only:
    - The system prompt (automatically injected by create_agent)
    - The last 2 conversation messages
    """
    messages = state["messages"]

    if len(messages) <= 2:
        return {"messages": messages}

    # Keep only last 2 non-system messages
    trimmed = messages[-2:]

    # Get back to last AIMessage if multiple tools are called one after another
    while trimmed and trimmed[0].type == "tool":
        start_idx = len(messages) - len(trimmed) - 1
        if start_idx < 0:
            break
        trimmed = messages[start_idx:]


    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed
        ]
    }


clear_tool_outputs = ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit()
    ]
)

model = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv('MISTRAL_API_KEY'))
agent = create_agent(
    model=model, 
    tools=tools, 
    # system_prompt=system_prompt, 
    state_schema=State, 
    context_schema=Context, 
    middleware=[trim_messages, clear_tool_outputs, dynamic_system_prompt],
    debug=True)