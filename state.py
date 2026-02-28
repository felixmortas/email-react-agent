from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated



class State(AgentState):
    messages: Annotated[list[BaseMessage], add_messages]