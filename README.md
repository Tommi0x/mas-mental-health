# MAS Clinical Diagnosis

Model-based multi-agent system for psychiatric clinical cases. Multiple LLM judges analyze the same free-text case independently; results are aggregated by majority, superconsent (75%), or Copeland score.

## Requirements

- Python 3.11+
- A free [Groq](https://console.groq.com) API key (no credit card required)

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy the example environment file and add your Groq key:

   ```bash
   copy .env.example .env
   ```

   Edit `.env` and set:

   ```
   GROQ_API_KEY=gsk_...
   ```

3. Verify connectivity:

   ```bash
   python test_llm.py
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Default agents

All agents use Groq's free tier via LiteLLM:

| Agent | Model |
|-------|-------|
| Judge_Llama70B | `groq/llama-3.3-70b-versatile` |
| Judge_Llama8B | `groq/llama-3.1-8b-instant` |
| Judge_Qwen32B | `groq/qwen/qwen3-32b` |
| Judge_Llama4Scout | `groq/meta-llama/llama-4-scout-17b-16e-instruct` |
| Judge_OSS20B | `groq/openai/gpt-oss-20b` |
| Judge_OSS120B | `groq/openai/gpt-oss-120b` |

## Project layout

- `app.py` — Streamlit UI
- `agents.py` — agent definitions and prompts
- `llm.py` — LiteLLM wrapper
- `graph.py` — LangGraph workflow and aggregation rules
- `aggregate.py` — majority, superconsent, and Copeland rules
- `models.py` — Pydantic data models
- `knowledge_base/` — DSM-5 reference sections for prompts

## Disclaimer

This is a research demo, not a clinical tool. Case text is sent to Groq's API when you run an analysis.
