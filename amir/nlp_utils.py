from __future__ import annotations

import logging
import os
import re
from itertools import islice

import torch
from transformers import pipeline

LOGGER = logging.getLogger(__name__)

WORD_RE = re.compile(r"\b[\w&.-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    return sentences or [text.strip()] if text.strip() else []


def keyword_density(text: str, keywords: list[str]) -> float:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    total_hits = sum(len(re.findall(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", text.lower())) for keyword in keywords)
    return total_hits / len(tokens)


def batched(items: list[str], size: int):
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def chunk_sentences(sentences: list[str], max_chars: int = 1000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        proposed = current_length + len(sentence) + (1 if current else 0)
        if current and proposed > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_length = len(sentence)
        else:
            current.append(sentence)
            current_length = proposed

    if current:
        chunks.append(" ".join(current))
    return chunks


def resolve_pipeline_device() -> int:
    if torch.cuda.is_available():
        return 0
    if os.environ.get("COLAB_TPU_ADDR"):
        LOGGER.warning(
            "A TPU runtime is detected, but this scorer currently uses the PyTorch/Transformers "
            "GPU path. Without torch-xla integration it will fall back to CPU."
        )
    return -1


class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 16, max_sentences: int | None = 160):
        self.batch_size = batch_size
        self.max_sentences = max_sentences
        self.device = resolve_pipeline_device()
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            top_k=None,
        )

    def summarize(self, text: str) -> dict[str, float]:
        sentences = split_sentences(text)
        if self.max_sentences is not None:
            sentences = sentences[: self.max_sentences]
        chunks = chunk_sentences(sentences)

        if not chunks:
            return {
                "sentiment_score": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "neutral_score": 0.0,
                "avg_confidence": 0.0,
                "segment_count": 0,
            }

        results = []
        for batch in batched(chunks, self.batch_size):
            results.extend(self.pipeline(batch, truncation=True, max_length=512))

        positive_scores: list[float] = []
        negative_scores: list[float] = []
        neutral_scores: list[float] = []
        confidences: list[float] = []

        for result in results:
            label_scores = {entry["label"].lower(): float(entry["score"]) for entry in result}
            positive_scores.append(label_scores.get("positive", 0.0))
            negative_scores.append(label_scores.get("negative", 0.0))
            neutral_scores.append(label_scores.get("neutral", 0.0))
            confidences.append(max(label_scores.values(), default=0.0))

        positive = sum(positive_scores) / len(positive_scores)
        negative = sum(negative_scores) / len(negative_scores)
        neutral = sum(neutral_scores) / len(neutral_scores)
        return {
            "sentiment_score": positive - negative,
            "positive_score": positive,
            "negative_score": negative,
            "neutral_score": neutral,
            "avg_confidence": sum(confidences) / len(confidences),
            "segment_count": len(chunks),
        }


class LoughranMcDonaldLexicon:
    POSITIVE_TERMS = {
        "achieve",
        "benefit",
        "confidence",
        "growth",
        "improve",
        "momentum",
        "opportunity",
        "outperform",
        "resilient",
        "strong",
    }
    NEGATIVE_TERMS = {
        "challenge",
        "decline",
        "decrease",
        "headwind",
        "pressure",
        "risk",
        "softness",
        "uncertain",
        "volatility",
        "weak",
    }
    UNCERTAINTY_TERMS = {"could", "may", "might", "uncertain", "unknown", "volatility"}

    def score(self, text: str) -> dict[str, float]:
        tokens = [token.lower() for token in WORD_RE.findall(text)]
        if not tokens:
            return {
                "lm_positive_density": 0.0,
                "lm_negative_density": 0.0,
                "lm_uncertainty_density": 0.0,
                "lm_net_tone": 0.0,
            }

        positive = sum(token in self.POSITIVE_TERMS for token in tokens)
        negative = sum(token in self.NEGATIVE_TERMS for token in tokens)
        uncertainty = sum(token in self.UNCERTAINTY_TERMS for token in tokens)
        total = len(tokens)
        return {
            "lm_positive_density": positive / total,
            "lm_negative_density": negative / total,
            "lm_uncertainty_density": uncertainty / total,
            "lm_net_tone": (positive - negative) / total,
        }
