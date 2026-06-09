import os
from pathlib import Path

from dotenv import load_dotenv, set_key

load_dotenv()

ENV_FILE = Path(__file__).resolve().parent / ".env"


# Return the environment variable name for the API key required by the model, if any.
def _env_name_for_model(model: str) -> str | None:
    if model.startswith("groq/"):
        return "GROQ_API_KEY"
    if model.startswith("gemini/"):
        return "GEMINI_API_KEY"
    return None


# Return setup hints for a missing API key environment variable.
def _missing_key_message(env_name: str) -> str:
    if env_name == "GROQ_API_KEY":
        return (
            "Groq API key is not set. Enter it in the sidebar, add GROQ_API_KEY to `.env`, "
            "or create a free key at https://console.groq.com"
        )
    if env_name == "GEMINI_API_KEY":
        return (
            "Gemini API key is not set. Enter it in the sidebar, add GEMINI_API_KEY to `.env`, "
            "or create a free key at https://aistudio.google.com/apikey"
        )
    return f"{env_name} is not set."


# Resolve an API key from the UI map first, then from the environment.
def resolve_api_key(env_name: str, api_keys: dict[str, str] | None = None) -> str:
    if api_keys:
        ui_key = api_keys.get(env_name, "").strip()
        if ui_key:
            return ui_key
    return os.environ.get(env_name, "").strip()


# Persist non-empty API keys to `.env` and update the current process environment.
def save_api_keys_to_env(api_keys: dict[str, str]) -> None:
    if not ENV_FILE.exists():
        ENV_FILE.write_text("", encoding="utf-8")
    for env_name, value in api_keys.items():
        key = value.strip()
        if not key:
            continue
        set_key(str(ENV_FILE), env_name, key)
        os.environ[env_name] = key


# Return whether the model's provider API key is configured.
def is_model_available(model: str, api_keys: dict[str, str] | None = None) -> bool:
    env_name = _env_name_for_model(model)
    if env_name is None:
        return True
    return bool(resolve_api_key(env_name, api_keys))


# Return error messages for models whose required API key is not configured.
def missing_api_key_errors(
    models: list[str], api_keys: dict[str, str] | None = None
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for model in models:
        env_name = _env_name_for_model(model)
        if env_name is None or env_name in seen:
            continue
        if not resolve_api_key(env_name, api_keys):
            seen.add(env_name)
            errors.append(_missing_key_message(env_name))
    return errors


# Return the API key required for the given LiteLLM model prefix, if any.
def _api_key_for_model(model: str, api_keys: dict[str, str] | None = None) -> str | None:
    env_name = _env_name_for_model(model)
    if env_name is None:
        return None
    key = resolve_api_key(env_name, api_keys)
    if not key:
        raise ValueError(_missing_key_message(env_name))
    return key


# Send the prompt to the given LiteLLM model and return the reply text.
def ask_llm(
    prompt: str, model: str, api_keys: dict[str, str] | None = None
) -> str:
    import litellm

    api_key = _api_key_for_model(model, api_keys)
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_key is not None:
        kwargs["api_key"] = api_key

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content
