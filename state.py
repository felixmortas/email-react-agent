from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated



class State(AgentState):
    messages: Annotated[list[BaseMessage], add_messages]
    isConnectionPageReached: bool = False
    isUsernameFilled: bool = False
    isPasswordFilled: bool = False
    isLogedIn: bool = False
    isUserProfilPageReached: bool = False
    isChangeEmailPageReached: bool = False
    isActualEmailFilled: bool = False
    isNewEmailFilled: bool = False
    isEmailChanged: bool = False
    