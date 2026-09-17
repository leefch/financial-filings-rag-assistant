"""Provider-agnostic chat LLM. Same shape as the rest of the stack so swapping
Claude for OpenAI is a one-line env change."""
from config import LLM_PROVIDER, ANTHROPIC_MODEL, OPENAI_MODEL

_llm = None


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _llm = ChatAnthropic(model=ANTHROPIC_MODEL,  max_tokens=1024)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    else:
        raise ValueError(f"unknown LLM_PROVIDER: {LLM_PROVIDER}")
    return _llm


def ask(system: str, user: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)
