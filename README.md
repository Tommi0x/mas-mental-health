# MAS Clinical Diagnosis

Model-based multi-agent system for psychiatric clinical cases. Multiple LLM judges analyze the same free-text case independently; results are aggregated by majority, superconsent, Copeland score, or weighted voting.

## Requirements

- Python 3.11+
- At least one free API key from a supported provider (no credit card required):
  - [Groq](https://console.groq.com)
  - [Gemini](https://aistudio.google.com/apikey)

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
| Gemini25Flash | `gemini/gemini-2.5-flash` |
| Gemini25FlashLite | `gemini/gemini-2.5-flash-lite` |
| Gemini3Flash | `gemini/gemini-3-flash-preview` |
| Gemini31FlashLite | `gemini/gemini-3.1-flash-lite` |
| Gemini35Flash | `gemini/gemini-3.5-flash` |

## Project layout

- `app.py` — Streamlit UI
- `agents.py` — agent definitions and prompts
- `llm.py` — LiteLLM wrapper
- `graph.py` — LangGraph workflow and aggregation rules
- `aggregate.py` — majority, superconsent, Copeland, and weighted rules
- `models.py` — Pydantic data models
- `knowledge_base/` — DSM-5 reference sections for prompts

## Disclaimer

This is a research demo, not a clinical tool. Case text is sent to the configured cloud APIs (Groq, Gemini) when you run an analysis.
