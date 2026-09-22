# D-Hub Research Agent

D-Hub is a dependency-free multi-agent research system. It decomposes a task, retrieves text from local files or web pages, ranks the sources, and produces an extractive report.

The workflow is hierarchical: the planner creates sub-tasks, the retriever gathers evidence, the analyzer evaluates every sub-task against that evidence, and the synthesizer produces the final report. Results can be printed, saved automatically, or emitted as structured JSON for another process to consume. The default implementation is deterministic and offline-friendly; the module boundaries are ready for an LLM-backed planner or summarizer when model credentials are available.

## Run it

```powershell
python main.py "What are the main benefits of renewable energy?" notes.txt https://example.com
```

Use `--json` for the task breakdown and source metadata, or `--output report.txt` to save the report. Local files must be UTF-8 text files.

## Docker

```powershell
docker compose run --rm dhub "Summarize this document" /data/notes.txt
```

Mount a local folder at `/data` when using Docker. The project intentionally has no mandatory third-party Python dependencies.
