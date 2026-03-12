from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware

fallback = ModelFallbackMiddleware(
    "mistralai:mistral-large-latest",
    "google_genai:gemini-3.1-flash-lite-preview",
    "google_genai:gemini-3-flash-preview",
    "mistralai:mistral-small-latest",
)
