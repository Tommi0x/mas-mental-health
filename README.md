# MAS Clinical Diagnosis

Model-based multi-agent system for psychiatric clinical cases. Multiple LLM judges analyze the same free-text case independently; results are aggregated by majority, superconsent, Copeland score, or weighted voting.

## Requirements

- Python 3.11+
- At least one free API key from a supported provider (no credit card required):
  - [Groq](https://console.groq.com)
  - [Gemini](https://aistudio.google.com/apikey)
  - [OpenRouter](https://openrouter.ai/keys)

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy the example environment file and add API keys for the providers you want to use:

   ```bash
   copy .env.example .env
   ```

   Edit `.env` and set any combination of:

   ```
   GROQ_API_KEY=gsk_...
   GEMINI_API_KEY=...
   OPENROUTER_API_KEY=sk-or-...
   ```

   Each key is optional; you only need keys for the providers whose agents you plan to run.

3. Verify connectivity:

   ```bash
   python test_llm.py
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Default agents

Agents use supported providers' free tiers via LiteLLM (configure one or more):

| Agent | Model |
|-------|-------|
| Llama70B | `groq/llama-3.3-70b-versatile` |
| Llama8B | `groq/llama-3.1-8b-instant` |
| Qwen32B | `groq/qwen/qwen3-32b` |
| Llama4Scout | `groq/meta-llama/llama-4-scout-17b-16e-instruct` |
| OSS20B | `groq/openai/gpt-oss-20b` |
| OSS120B | `groq/openai/gpt-oss-120b` |
| Gemini31FlashLite | `gemini/gemini-3.1-flash-lite` |
| Gemini35Flash | `gemini/gemini-3.5-flash` |

### OpenRouter free agents (`:free` tier)

Requires `OPENROUTER_API_KEY`. Overlaps with Groq/Gemini agents are omitted; one model per family on OpenRouter.

| Agent | Model |
|-------|-------|
| OR_DolphinMistral24B | `openrouter/cognitivecomputations/dolphin-mistral-24b-venice-edition:free` |
| OR_Gemma4_26B | `openrouter/google/gemma-4-26b-a4b-it:free` |
| OR_KimiK26 | `openrouter/moonshotai/kimi-k2.6:free` |
| OR_NemotronNano9B | `openrouter/nvidia/nemotron-nano-9b-v2:free` |
| OR_LagunaXS2 | `openrouter/poolside/laguna-xs.2:free` |
| OR_NexN2Pro | `openrouter/nex-agi/nex-n2-pro:free` |

## Project layout

- `app.py` — Streamlit UI
- `agents.py` — agent definitions and prompts
- `llm.py` — LiteLLM wrapper
- `graph.py` — LangGraph workflow and aggregation rules
- `aggregate.py` — majority, superconsent, Copeland, and weighted rules
- `models.py` — Pydantic data models
- `knowledge_base/` — DSM-5 reference sections for prompts

## Disclaimer

This is a research demo, not a clinical tool. Case text is sent to the configured cloud APIs when you run an analysis.
