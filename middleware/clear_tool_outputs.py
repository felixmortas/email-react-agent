from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

clear_tool_outputs = ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit()
    ]
)
