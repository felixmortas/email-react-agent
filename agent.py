import os

from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import tools
from middleware.dynamic_system_prompt import dynamic_system_prompt
from middleware.trim_messages import make_trim_messages
from middleware.clear_tool_outputs import clear_tool_outputs
from middleware.model_fallback import fallback
from state import State
from context import Context

def create_email_agent(model_name: str = "mistral-large-latest"):
    if model_name.startswith("gemini"):
        model = ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv('GOOGLE_API_KEY')) # gemini-3-flash-preview gemini-2.5-pro gemini-2.5-flash gemini-2.5-flash-lite gemini-2.5-flash-lite-preview-09-2025 gemini-2.5-flash-native-audio-preview-12-2025 gemini-2.5-flash-preview-tts gemini-2.0-flash gemini-2.0-flash-lite
        trim_messages = make_trim_messages(is_gemini=True)

    elif model_name.startswith("mistral"):
        model = ChatMistralAI(model=model_name, api_key=os.getenv('MISTRAL_API_KEY'))
        trim_messages = make_trim_messages(is_gemini=False)

    else:
        raise ValueError(f"Modèle non supporté : {model_name}. Utilisez un modèle 'gemini-*' ou 'mistral-*'.")

    return create_agent(
        model=model, 
        tools=tools, 
        state_schema=State, 
        context_schema=Context, 
        middleware=[trim_messages, clear_tool_outputs, dynamic_system_prompt, fallback],
        debug=True
    )