from langfuse_engine import langfuse
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from step import Step

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    state = request.state
    raw = state.get("step", Step.FIND_LOGIN_PAGE)
    current_step = raw.value if isinstance(raw, Step) else raw
    current_url = state.get("current_url")
    system_prompt = langfuse.get_prompt(f"fr/{current_step}", type="text")

    compiled_prompt = system_prompt.compile(
        current_step=current_step,
        current_url=current_url,
    )

    return compiled_prompt


@dynamic_prompt
def dynamic_system_prompt_eco(request: ModelRequest) -> str:
    state = request.state
    current_url = state.get("current_url")
    last_step_url = state.get("last_step_url")
    isConnectionPageReached = state.get('isConnectionPageReached', False)
    isLogedIn = state.get('isLogedIn', False)
    isChangeEmailPageReached = state.get('isChangeEmailPageReached', False)
    isEmailChanged = state.get('isEmailChanged', False)
    
    system_prompt = langfuse.get_prompt("fr/eco-email-change", type="text")

    compiled_prompt = system_prompt.compile(
        current_url=current_url,
        last_step_url=last_step_url,
        isConnectionPageReached=isConnectionPageReached,
        isLogedIn=isLogedIn,
        isChangeEmailPageReached=isChangeEmailPageReached,
        isEmailChanged=isEmailChanged,
    )

    return compiled_prompt