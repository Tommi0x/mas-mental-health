from llm import ask_llm

TEST_MODELS = [
    "ollama/llama3.2",
    "ollama/mistral",
    "ollama/phi3",
    "ollama/gemma2:2b",
]

TEST_PROMPT = "Reply with exactly one short sentence confirming you received this test."


# Run a short test prompt against each model in TEST_MODELS and print the replies.
def main() -> None:
    for model in TEST_MODELS:
        print(f"\n--- {model} ---")
        try:
            text = ask_llm(TEST_PROMPT, model)
            print(text)
        except Exception as exc:
            print(f"ERROR: {exc}")
            raise


if __name__ == "__main__":
    main()
