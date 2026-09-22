"""Deterministic extractive summarization for retrieved documents."""

import re
from typing import Iterable, List

from retriever import Document


def _sentences(text: str) -> List[str]:
	return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def summarize(text: str, query: str = "", max_sentences: int = 5) -> str:
	"""Select the most relevant sentences while keeping source order."""
	sentences = _sentences(text)
	if len(sentences) <= max_sentences:
		return " ".join(sentences)
	terms = {term.lower() for term in query.split() if len(term) > 2}
	scored = [(sum(sentence.lower().count(term) for term in terms), index, sentence)
			  for index, sentence in enumerate(sentences)]
	chosen = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda item: item[1])
	return " ".join(sentence for _, _, sentence in chosen)


def build_report(task: str, documents: Iterable[Document], max_sentences: int = 5,
				 subtasks: Iterable[str] = ()) -> str:
	sections = [f"Research task: {task}", "", "Findings"]
	document_list = list(documents)
	if subtasks:
		sections.extend(["", "Subtask analysis"])
		combined_text = " ".join(document.text for document in document_list)
		for subtask in subtasks:
			finding = summarize(combined_text, subtask, 2) or "No evidence matched this subtask."
			sections.extend([f"\n{subtask}", finding])
	for document in document_list:
		sections.extend([f"\nSource: {document.source}", summarize(document.text, task, max_sentences) or "No readable text found."])
	if len(sections) == 3:
		sections.append("No documents were retrieved.")
	return "\n".join(sections)
