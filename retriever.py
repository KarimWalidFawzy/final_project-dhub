"""Retrieve text from local files and web pages without third-party packages."""

import csv
import io
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen
from transformers import pipeline, Pipeline,AutoTokenizer, AutoModelForSeq2SeqLM

@dataclass
class Document:
	source: str
	text: str


class _TextParser(HTMLParser):
	def __init__(self) -> None:
		super().__init__()
		self.parts: List[str] = []
		self._ignored = 0

	def handle_starttag(self, tag: str, attrs: object) -> None:
		if tag in {"script", "style", "noscript"}:
			self._ignored += 1

	def handle_endtag(self, tag: str) -> None:
		if tag in {"script", "style", "noscript"} and self._ignored:
			self._ignored -= 1

	def handle_data(self, data: str) -> None:
		if not self._ignored and data.strip():
			self.parts.append(data.strip())


def _clean(text: str) -> str:
	return " ".join(text.split())


def _pdf_text(path: Path) -> str:
	try:
		from pypdf import PdfReader
	except ImportError as exc:
		raise RuntimeError("PDF support requires pypdf; install requirements.txt") from exc
	return " ".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _table_text(rows: Iterable[object]) -> str:
	return " ".join(" ".join(str(value) for value in row.values()) if isinstance(row, dict)
					 else " ".join(str(value) for value in row) if isinstance(row, (list, tuple))
					 else str(row) for row in rows)


def _sqlite_text(path: Path) -> str:
	with sqlite3.connect(path) as connection:
		tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
		parts: List[str] = []
		for (table,) in tables:
			rows = connection.execute(f'SELECT * FROM "{table}" LIMIT 100').fetchall()
			parts.append(f"Table {table}: {_table_text(rows)}")
		return " ".join(parts)


def _load_local(path: Path) -> str:
	suffix = path.suffix.lower()
	if suffix == ".pdf":
		return _clean(_pdf_text(path))
	if suffix in {".json", ".jsonl"}:
		return _clean(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=True))
	if suffix == ".csv":
		with path.open(newline="", encoding="utf-8") as handle:
			return _clean(_table_text(csv.DictReader(handle)))
	if suffix in {".db", ".sqlite", ".sqlite3"}:
		return _clean(_sqlite_text(path))
	return _clean(path.read_text(encoding="utf-8"))


def _google_sheet_url(source: str) -> str:
	if "docs.google.com/spreadsheets" in source and "format=" not in source:
		return source.split("/edit")[0] + "/export?format=csv"
	return source


def load_source(source: str, timeout: int = 12) -> Document:
	"""Load text, PDF, CSV, JSON, SQLite, API, or public Google Sheet data."""
	if source.startswith(("http://", "https://")):
		source = _google_sheet_url(source)
		request = Request(source, headers={"User-Agent": "D-Hub-Research-Agent/1.0"})
		with urlopen(request, timeout=timeout) as response:
			raw = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
		content_type = response.headers.get_content_type()
		if "json" in content_type or source.lower().split("?")[0].endswith(".json"):
			try:
				return Document(source, _clean(json.dumps(json.loads(raw), ensure_ascii=True)))
			except json.JSONDecodeError:
				pass
		if "csv" in content_type or "format=csv" in source:
			return Document(source, _clean(_table_text(csv.DictReader(io.StringIO(raw)))))
		parser = _TextParser()
		parser.feed(raw)
		return Document(source, _clean(" ".join(parser.parts)))
	path = Path(source).expanduser()
	return Document(str(path), _load_local(path))


def retrieve(sources: Iterable[str], query: str = "") -> List[Document]:
	"""Load sources and rank them by the number of query terms they contain."""
	terms = {term.lower() for term in query.split() if len(term) > 2}
	documents: List[Document] = []
	errors: List[str] = []
	with ThreadPoolExecutor(max_workers=min(8, max(1, len(list(sources))))) as pool:
		futures = {pool.submit(load_source, source): source for source in sources}
		for future in as_completed(futures):
			source = futures[future]
			try:
				document = future.result()
				if document.text:
					documents.append(document)
			except (OSError, ValueError, TimeoutError, RuntimeError) as exc:
				errors.append(f"{source}: {exc}")
	documents.sort(key=lambda item: sum(item.text.lower().count(term) for term in terms), reverse=True)
	if errors and not documents:
		raise RuntimeError("No sources could be loaded: " + "; ".join(errors))
	return documents
