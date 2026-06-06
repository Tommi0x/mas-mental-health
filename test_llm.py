from agents import AGENTS
from llm import ask_llm

TEST_PROMPT = "Reply with exactly one short sentence confirming you received this test."


# Run a short test prompt against each configured agent model and print the replies.
def main() -> None:
    for agent in AGENTS:
        model = agent["model"]
        print(f"\n--- {agent['name']} ({model}) ---")
        try:
            text = ask_llm(TEST_PROMPT, model)
            print(text)
        except Exception as exc:
            print(f"ERROR: {exc}")
            raise


if __name__ == "__main__":
    main()
