"""The RAG chain: retrieve -> gate -> generate -> verify -> cite.

Built with LCEL (`prompt | llm | parser`). Two guardrails bracket generation:

* a **relevance gate** on the *final* retrieval set runs before the chain, so an
  out-of-scope question is refused without ever reaching the LLM; and
* a **groundedness check** after generation rejects any answer whose citations
  do not point at real retrieved chunks.

Neither depends on the model's good behaviour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .config import Settings, settings
from .retrieval import (
    format_citations,
    format_context,
    hybrid_retrieve,
    is_relevant,
    retrieve_with_scores,
    score_documents,
)

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "I don't have enough information."
UNVERIFIED_MESSAGE = "I couldn't verify the answer using the retrieved documents."

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You are a support assistant for Electro Pi. Follow these rules exactly:\n"
    "1. Answer using ONLY the numbered context provided. Never use outside or "
    "prior knowledge.\n"
    "2. Every sentence that states a fact MUST end with a bracketed citation to "
    "the numbered chunk it came from, e.g. [1] or [2]. An answer with no citation "
    "is invalid.\n"
    "3. Only cite numbers that actually appear in the context. Never invent a "
    "citation or cite a number that is not shown.\n"
    "4. If the context does not contain the answer, reply with exactly: "
    f"{NO_CONTEXT_MESSAGE}\n"
    "5. Do not guess. If you are uncertain, reply with that exact sentence.\n"
    "Be concise."
)

# A one-shot exchange pins the citation format for small instruction-tuned
# models, which otherwise tend to answer fluently but omit the brackets.
_EXAMPLE_CONTEXT = (
    "[1] returns_policy.md — Return window\n"
    "Development boards and kits have an extended window of 30 calendar days."
)
_EXAMPLE_QUESTION = "How long do I have to return a development board?"
_EXAMPLE_ANSWER = "Development boards can be returned within 30 calendar days [1]."

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{example_context}\n\nQuestion: {example_question}"),
        ("ai", "{example_answer}"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
).partial(
    example_context=_EXAMPLE_CONTEXT,
    example_question=_EXAMPLE_QUESTION,
    example_answer=_EXAMPLE_ANSWER,
)


@dataclass
class RAGResponse:
    """Result of one RAG query."""

    question: str
    answer: str
    citations: list[str]
    contexts: list[str]
    refused: bool
    grounded: bool = True

    def to_markdown(self) -> str:
        """Render the response for the outputs report."""
        lines = [f"### Q: {self.question}", "", f"**Answer:** {self.answer}", ""]
        if self.refused:
            lines.append("_Refused by the relevance gate; the LLM was not called._")
            return "\n".join(lines)
        if not self.grounded:
            lines.append(
                "_Answer failed groundedness validation; its citations did not "
                "match the retrieved context, so it was rejected._"
            )
            return "\n".join(lines)
        lines.append("**Citations:**")
        lines += [f"- {c}" for c in self.citations] or ["- (none)"]
        lines += ["", "<details><summary>Retrieved context</summary>", ""]
        lines += [f"```\n{ctx}\n```" for ctx in self.contexts]
        lines += ["", "</details>"]
        return "\n".join(lines)


def _is_self_refusal(answer: str) -> bool:
    """True if the model returned the fixed no-context refusal string."""
    return answer.lower().startswith("i don't have enough information")


def build_llm(cfg: Settings = settings) -> ChatOllama:
    """Construct the local Ollama chat model used for generation."""
    logger.info("LLM: Ollama %s at %s", cfg.ollama_model, cfg.ollama_url)
    return ChatOllama(
        model=cfg.ollama_model,
        base_url=cfg.ollama_url,
        temperature=cfg.temperature,
    )


def answer_question(
    store,
    question: str,
    all_chunks: list[Document] | None = None,
    cfg: Settings = settings,
) -> RAGResponse:
    """Answer a question against the indexed corpus.

    Args:
        store: The Chroma vector store.
        question: The user's question.
        all_chunks: Full chunk list; required when hybrid search is enabled.
        cfg: Settings.

    Returns:
        A :class:`RAGResponse`. If retrieval finds nothing relevant, the answer
        is the fixed refusal string and ``refused`` is True. If the generated
        answer's citations cannot be verified against the retrieved chunks, the
        answer is the unverified-message and ``grounded`` is False.
    """
    # 1. Retrieve first. Hybrid (BM25 + dense) when a chunk set is available,
    #    otherwise dense-only. This is the evidence the gate and answer share.
    if cfg.use_hybrid_search and all_chunks:
        docs = hybrid_retrieve(store, all_chunks, question, cfg)
    else:
        docs = [doc for doc, _ in retrieve_with_scores(store, question, cfg)]

    # 2. Guardrail: score the FINAL retrieval set and refuse deterministically
    #    before involving the LLM if nothing clears the relevance threshold.
    scored = score_documents(store, question, docs, cfg)
    if not is_relevant(scored, cfg):
        logger.info("Refusing out-of-scope question: %r", question)
        return RAGResponse(
            question=question,
            answer=NO_CONTEXT_MESSAGE,
            citations=[],
            contexts=[],
            refused=True,
        )

    # 3. Generate. Build the chain once; it may be invoked twice (step 4).
    context = format_context(docs)
    chain = PROMPT | build_llm(cfg) | StrOutputParser()

    def _generate(user_question: str) -> str:
        try:
            return chain.invoke(
                {"context": context, "question": user_question}
            ).strip()
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise RuntimeError(
                f"Generation failed -- is Ollama running at {cfg.ollama_url}? ({exc})"
            ) from exc

    answer = _generate(question)

    # 4. If the model answered but forgot to cite, give it one corrective retry
    #    before the groundedness check rejects an otherwise-valid answer -- a
    #    small model often omits brackets it was capable of producing. This does
    #    not weaken the guard: a hallucinated citation is still rejected below.
    if not _is_self_refusal(answer) and not _CITATION_RE.search(answer):
        logger.info("Answer had no citations; retrying once with a reminder")
        retry = _generate(
            f"{question}\n\nReminder: end every factual sentence with the "
            "bracketed number of the chunk it came from, e.g. [1]."
        )
        if not _is_self_refusal(retry) and _CITATION_RE.search(retry):
            answer = retry

    # 5. The model may still self-refuse; honour that as a gate refusal.
    if _is_self_refusal(answer):
        return RAGResponse(
            question=question,
            answer=NO_CONTEXT_MESSAGE,
            citations=[],
            contexts=[],
            refused=True,
        )

    # 6. Groundedness check: keep only citations that point at real chunks, and
    #    reject the answer outright if it cites anything that does not exist or
    #    cites nothing at all. A hallucinated [5] must never reach the user.
    cited = {int(n) for n in _CITATION_RE.findall(answer)}
    valid = {n for n in cited if 1 <= n <= len(docs)}
    invalid = cited - valid
    if invalid or not valid:
        logger.warning(
            "Groundedness check failed for %r: cited=%s valid=%s invalid=%s",
            question,
            sorted(cited),
            sorted(valid),
            sorted(invalid),
        )
        return RAGResponse(
            question=question,
            answer=UNVERIFIED_MESSAGE,
            citations=[],
            contexts=[],
            refused=False,
            grounded=False,
        )

    return RAGResponse(
        question=question,
        answer=answer,
        citations=format_citations(docs, referenced=valid),
        contexts=[context],
        refused=False,
        grounded=True,
    )
