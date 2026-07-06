"""Shared utilities for data preprocessing scripts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BERTWEET_EMOTION_MODEL = "finiteautomata/bertweet-base-emotion-analysis"

TEXT_EMOTION_KEYS = [
    "anger",
    "joy",
    "fear",
    "sadness",
    "disgust",
    "surprise",
    "neutral",
]

BERTWEET_TO_TEXT_EMOTION = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "surprise",
    "others": "neutral",
}


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json_or_jsonl(path: str | os.PathLike[str], input_format: str = "auto") -> list[dict[str, Any]]:
    """Read either a JSON list file or a JSONL file."""
    path = Path(path)
    if input_format == "auto":
        input_format = "jsonl" if path.suffix.lower() == ".jsonl" else "json"

    if input_format == "jsonl":
        return list(read_jsonl(path))
    if input_format == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON list in {path}, but got {type(data).__name__}.")

    raise ValueError("input_format must be one of: auto, json, jsonl")


def write_jsonl(records: Iterable[dict[str, Any]], path: str | os.PathLike[str], mode: str = "w") -> None:
    """Write records to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(records: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    """Append records to a CSV file and write the header only once."""
    if not records:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def load_existing_ids(csv_path: str | os.PathLike[str], id_column: str) -> set[str]:
    """Load processed ids from an existing CSV file for resumable preprocessing."""
    path = Path(csv_path)
    if not path.exists():
        return set()

    done: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if id_column not in (reader.fieldnames or []):
            return done
        for row in reader:
            value = row.get(id_column)
            if value is not None:
                done.add(str(value))
    return done


def map_binary_label(value: Any) -> Optional[int]:
    """Map common binary fake-news labels to 0=fake and 1=real."""
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()
    fake_values = {"0", "fake", "false", "misleading", "manipulated"}
    real_values = {"1", "real", "true", "original", "genuine"}

    if text in fake_values:
        return 0
    if text in real_values:
        return 1
    return None


class BertweetEmotionAnalyzer:
    """BERTweet-based emotion analyzer with a fixed 7-dimensional output space."""

    def __init__(
        self,
        model_name: str = BERTWEET_EMOTION_MODEL,
        device: Optional[str] = None,
        max_text_chars: int = 512,
        max_tokens: int = 128,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_text_chars = max_text_chars
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, float]:
        """Return emotion probabilities using the repository's emotion label order."""
        text = text if isinstance(text, str) else ""
        text = text.strip()
        if not text:
            return {label: 0.0 for label in TEXT_EMOTION_KEYS}

        encoded = self.tokenizer(
            text[: self.max_text_chars],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(0)
            probabilities = F.softmax(logits, dim=-1).detach().cpu().tolist()

        scores = {label: 0.0 for label in TEXT_EMOTION_KEYS}
        for index, probability in enumerate(probabilities):
            source_label = str(self.model.config.id2label[index]).lower()
            target_label = BERTWEET_TO_TEXT_EMOTION.get(source_label)
            if target_label is not None:
                scores[target_label] += float(probability)
        return scores
