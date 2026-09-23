# D-Hub Research Agent

D-Hub is a multi-agent research system. It decomposes a task, retrieves text from local files or web pages, ranks the sources, and produces a report.

The workflow is hierarchical: the planner creates sub-tasks, the retriever gathers evidence, the analyzer evaluates every sub-task against that evidence, and the synthesizer produces the final report. When `GROQ_API_KEY` is configured, the analyzer and synthesizer use Groq automatically. Without credentials, the deterministic extractive fallback remains available.

## Run it

```powershell
python main.py "What are the main benefits of renewable energy?" notes.txt https://example.com
```

Set `GROQ_API_KEY` to enable Groq-backed multi-agent summaries. The model defaults to `llama-3.3-70b-versatile`; override it with `GROQ_MODEL`. Set `LLM_PROVIDER=extractive` to force offline behavior.

Use `--json` for the task breakdown and source metadata, or `--output report.txt` to save the report. Local files must be UTF-8 text files.

## Docker

```powershell
docker compose run --rm dhub "Summarize this document" /data/notes.txt
```

Mount a local folder at `/data` when using Docker. The project intentionally has no mandatory third-party Python dependencies.
