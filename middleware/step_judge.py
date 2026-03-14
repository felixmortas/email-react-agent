from langchain_core.messages import ToolMessage
from browser_helpers import extract_semantic_html
from langfuse_engine import langfuse

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import before_model
from langchain.tools import ToolRuntime

from context import Context
from state import State
from step import Step

async def _get_judge_model(model: str = "mistral-small-latest", model_provider: str = "mistralai"):
    model = init_chat_model(model=model, model_provider=model_provider)
    return model

async def _invoke_step_judge(
    steps_list: str,
    current_step: str,
    current_url: str,
    html_content: str,
) -> str | None:
    """Appelle un LLM léger pour classifier l'étape courante."""
    prompt = langfuse.get_prompt("fr/step-classifier", type="text")
    compiled = prompt.compile(
        steps_list=steps_list,
        current_step=current_step,
        current_url=current_url,
        html_content=html_content,
    )

    # Utiliser un modèle léger et rapide (pas besoin d'un Sonnet ici)
    judge_model = await _get_judge_model()
    response = await judge_model.ainvoke(compiled)
    detected_step = response.content.strip()

    valid_steps = {s.value for s in Step}
    if detected_step not in valid_steps:
        return None 

    return detected_step


@before_model
async def judge_current_step(state: State, runtime: ToolRuntime[Context]) -> dict:
    """
    Avant chaque appel LLM principal :
    - Demande au judge de classifier l'étape
    - Met à jour state.step si changement détecté
    """

    page = runtime.context["page"]
    steps_list = Step.values_list()
    raw = state.get("step", Step.FIND_LOGIN_PAGE)
    current_step = raw.value if isinstance(raw, Step) else raw
    current_url = page.url
    html_content = await extract_semantic_html(page)

    detected_step = await _invoke_step_judge(
        steps_list=steps_list,
        current_step=current_step,
        current_url=current_url,
        html_content=html_content,
    )

    if detected_step is None or detected_step == current_step:
        return {}  # pas de changement

    return {"step": detected_step}