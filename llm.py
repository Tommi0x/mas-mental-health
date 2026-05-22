# Send the prompt to the given LiteLLM model and return the reply text.
def ask_llm(prompt: str, model: str) -> str:
    import litellm

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
