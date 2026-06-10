import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv, set_key

load_dotenv()

ENV_FILE = Path(__file__).resolve().parent / ".env"


class ApiProvider(TypedDict):
    env_name: str
    model_prefix: str
    display_name: str
    signup_url: str


API_PROVIDERS: list[ApiProvider] = [
    {
        "env_name": "GROQ_API_KEY",
        "model_prefix": "groq/",
        "display_name": "Groq",
        "signup_url": "https://console.groq.com",
    },
    {
        "env_name": "GEMINI_API_KEY",
        "model_prefix": "gemini/",
        "display_name": "Gemini",
        "signup_url": "https://aistudio.google.com/apikey",
    },
    {
        "env_name": "OPENROUTER_API_KEY",
        "model_prefix": "openrouter/",
        "display_name": "OpenRouter",
        "signup_url": "https://openrouter.ai/keys",
    },
]


# Return the sidebar session-state key for a provider environment variable.
def provider_session_key(env_name: str) -> str:
    return env_name.replace("_API_KEY", "_api_key").lower()


# Return provider metadata for the given environment variable name.
def _provider_for_env(env_name: str) -> ApiProvider | None:
    for provider in API_PROVIDERS:
        if provider["env_name"] == env_name:
            return provider
    return None


# Return the environment variable name for the API key required by the model, if any.
def _env_name_for_model(model: str) -> str | None:
    for provider in API_PROVIDERS:
        if model.startswith(provider["model_prefix"]):
            return provider["env_name"]
    return None


# Return the model prefix that identifies the API provider for a LiteLLM model id.
def provider_key_for_model(model: str) -> str:
    for provider in API_PROVIDERS:
        if model.startswith(provider["model_prefix"]):
            return provider["model_prefix"]
    return "other"


# Return setup hints for a missing API key environment variable.
def _missing_key_message(env_name: str) -> str:
    provider = _provider_for_env(env_name)
    if provider is None:
        return f"{env_name} is not set."
    return (
        f"{provider['display_name']} API key is not set. Enter it in the sidebar, "
        f"add {env_name} to `.env`, or create a free key at {provider['signup_url']}"
    )


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
