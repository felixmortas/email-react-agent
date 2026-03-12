from langchain.messages import SystemMessage
from langfuse_engine import langfuse
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    state = request.state
    current_url = state.get("current_url")
    last_step_url = state.get("last_step_url")
    system_prompt = langfuse.get_prompt("email-change", type="text")

    compiled_prompt = system_prompt.compile(
        current_url=current_url,
        last_step_url=last_step_url,
        isConnectionPageReached=state.get('isConnectionPageReached', False),
        isLogedIn=state.get('isLogedIn', False),
        isChangeEmailPageReached=state.get('isChangeEmailPageReached', False),
        isEmailChanged=state.get('isEmailChanged', False),
    )

    return compiled_prompt