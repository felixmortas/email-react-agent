from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, ContextEditingMiddleware, ClearToolUsesEdit
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from tools import tools
from system_prompt import dynamic_system_prompt, system_prompt
from state import State
from context import Context

import os
from dotenv import load_dotenv
load_dotenv()

def make_trim_messages(is_gemini: bool = False):
    # Middleware to keep only the last 2 messages
    @before_model
    def _trim_messages(state: State, runtime: ToolRuntime[Context]):
        """
        Keep only:
        - The system prompt (automatically injected by create_agent)
        - The last 2 conversation messages if Mistral AI
        - The last complete exchange (HumanMessage → AIMessage → ToolMessage(s)) if Google Gemini
        """
        messages = state["messages"]

        if len(messages) <= 2:
            return {"messages": messages}

        # Start from the last 2 messages
        trimmed = messages[-2:]

        # Walk back to include the full AIMessage group (multiple tool responses)
        while trimmed and trimmed[0].type == "tool":
            start_idx = len(messages) - len(trimmed) - 1
            if start_idx < 0:
                break
            trimmed = messages[start_idx:]

        # Gemini requires that an AIMessage with tool_calls is preceded by a HumanMessage
        if is_gemini and trimmed and trimmed[0].type != "human":
            trimmed = [HumanMessage(content="."), *trimmed]


        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *trimmed
            ]
        }
    return _trim_messages


clear_tool_outputs = ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit()
    ]
)

# model = ChatMistralAI(model="mistral-large-latest", api_key=os.getenv('MISTRAL_API_KEY'))
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=os.getenv('GOOGLE_API_KEY')) # gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite gemini-2.5-flash-lite-preview-09-2025 gemini-2.5-flash-native-audio-preview-12-2025 gemini-2.5-flash-preview-tts gemini-2.0-flash gemini-2.0-flash-lite

is_gemini = "gemini" in model.model.lower()
trim_messages = make_trim_messages(is_gemini=is_gemini)

agent = create_agent(
    model=model, 
    tools=tools, 
    # system_prompt=system_prompt, 
    state_schema=State, 
    context_schema=Context, 
    middleware=[trim_messages, clear_tool_outputs, dynamic_system_prompt],
    debug=True)