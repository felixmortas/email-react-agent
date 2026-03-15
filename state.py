from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from step import Step

# Avoid errors when 2 tools are called in the same step and edit the same state value
def take_last(existing, new):
    return new

class State(AgentState):
    messages: Annotated[list[BaseMessage], add_messages]
    step: Step = Step.FIND_LOGIN_PAGE
    current_url: Annotated[str, take_last]
    last_step_url: str
    isConnectionPageReached: bool = False
    isLogedIn: bool = False
    isChangeEmailPageReached: bool = False
    isEmailChanged: bool = False
    