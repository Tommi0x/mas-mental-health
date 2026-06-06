import os

from dotenv import load_dotenv

load_dotenv()


# Return the API key required for the given LiteLLM model prefix, if any.
def _api_key_for_model(model: str) -> str | None:
    if model.startswith("groq/"):
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Create a free key at https://console.groq.com "
                "and add it to a .env file or your environment."
            )
        return key
    return None


# Send the prompt to the given LiteLLM model and return the reply text.
def ask_llm(prompt: str, model: str) -> str:
    import litellm

    api_key = _api_key_for_model(model)
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_key is not None:
        kwargs["api_key"] = api_key

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content
