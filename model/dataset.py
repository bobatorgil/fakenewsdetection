"""Dataset and feature-loading utilities for the Fakeddit experiments."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTImageProcessor

from .utils import parse_json_object

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEXT_EMOTION_KEYS = [
    "anger",
    "joy",
    "fear",
    "sadness",
    "disgust",
    "surprise",
    "neutral",
]

IMAGE_EMOTION_KEYS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
    "neutral",
]


class FakedditMultimodalDataset(Dataset):
    """Fakeddit dataset with text, image, comment, and emotion features.

    Each sample returns tokenized news text, tokenized comments, ViT image
    pixels, text emotion probabilities, image emotion probabilities,
    aggregated comment emotion statistics, and the binary fake-news label.
    """

    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        image_emotions_csv: str,
        text_emotions_csv: str,
        tokenizer: BertTokenizer,
        image_processor: ViTImageProcessor,
        max_text_length: int = 128,
        comment_emotions_csv: Optional[str] = None,
        comments_jsonl: Optional[str] = None,
        max_comment_length: int = 256,
        label_column: str = "2_way_label",
    ) -> None:
        self.data = pd.read_json(jsonl_path, lines=True)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.max_comment_length = max_comment_length
        self.label_column = label_column

        self.comment_text_map = self._load_comment_texts(comments_jsonl)
        self.comment_emotion_map = self._load_comment_emotions(comment_emotions_csv)
        self.image_emotion_map = self._load_image_emotions(image_emotions_csv)
        self.text_emotion_map = self._load_text_emotions(text_emotions_csv)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        sample_id = str(row["id"])
        text = str(row.get("text", ""))
        label = int(row[self.label_column])

        image = Image.open(self._resolve_image_path(sample_id)).convert("RGB")
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        comment_encoding = self.tokenizer(
            self.comment_text_map.get(sample_id, ""),
            truncation=True,
            padding="max_length",
            max_length=self.max_comment_length,
            return_tensors="pt",
        )
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        text_emotion = self.text_emotion_map.get(
            sample_id, np.zeros(len(TEXT_EMOTION_KEYS), dtype=np.float32)
        )
        image_emotion = self.image_emotion_map.get(
            sample_id, np.zeros(len(IMAGE_EMOTION_KEYS), dtype=np.float32)
        )
        comment_emotion = self.comment_emotion_map.get(
            sample_id, np.zeros(len(TEXT_EMOTION_KEYS) * 2, dtype=np.float32)
        )

        return {
            "input_ids_text": text_encoding["input_ids"].squeeze(0),
            "attention_text": text_encoding["attention_mask"].squeeze(0),
            "input_ids_comment": comment_encoding["input_ids"].squeeze(0),
            "attention_comment": comment_encoding["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "emotion_text": torch.from_numpy(text_emotion),
            "emotion_image": torch.from_numpy(image_emotion),
            "emotion_comment_stats": torch.from_numpy(comment_emotion),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _resolve_image_path(self, sample_id: str) -> Path:
        """Find the image file for a sample using common image extensions."""
        for extension in (".jpg", ".jpeg", ".png", ".webp"):
            image_path = self.images_dir / f"{sample_id}{extension}"
            if image_path.exists():
                return image_path
        raise FileNotFoundError(f"No image file found for sample id: {sample_id}")

    @staticmethod
    def _load_comment_texts(comments_jsonl: Optional[str]) -> dict[str, str]:
        """Load and merge comments by news id.

        Supported formats:
        1. One record per news item: {"news_id": ..., "comments": [{"comment": ...}, ...]}
        2. One record per comment: {"news_id": ..., "comment": ...}
        """
        if comments_jsonl is None or not os.path.exists(comments_jsonl):
            print(f"[WARN] comments_jsonl not found or not provided: {comments_jsonl}")
            return {}

        grouped_comments: dict[str, list[str]] = defaultdict(list)
        with open(comments_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                news_id = str(obj.get("news_id", obj.get("id", "")))
                if not news_id:
                    continue

                if isinstance(obj.get("comments"), list):
                    for comment_obj in obj["comments"]:
                        comment = str(comment_obj.get("comment", "")).strip()
                        if comment:
                            grouped_comments[news_id].append(comment)
                else:
                    comment = str(obj.get("comment", "")).strip()
                    if comment:
                        grouped_comments[news_id].append(comment)

        return {news_id: " [SEP] ".join(comments) for news_id, comments in grouped_comments.items()}

    @staticmethod
    def _load_comment_emotions(comment_emotions_csv: Optional[str]) -> dict[str, np.ndarray]:
        """Load comment emotion vectors and aggregate them into mean and standard deviation."""
        if comment_emotions_csv is None or not os.path.exists(comment_emotions_csv):
            print(f"[WARN] comment_emotions_csv not found or not provided: {comment_emotions_csv}")
            return {}

        df = pd.read_csv(comment_emotions_csv)
        if "news_id" not in df.columns or "emotion" not in df.columns:
            raise ValueError("comment_emotions_csv must contain 'news_id' and 'emotion' columns.")

        emotion_map: dict[str, np.ndarray] = {}
        df["emotion_vector"] = df["emotion"].apply(
            lambda value: np.array(
                [float(parse_json_object(value).get(key, 0.0)) for key in TEXT_EMOTION_KEYS],
                dtype=np.float32,
            )
        )

        for news_id, group in df.groupby("news_id"):
            matrix = np.stack(group["emotion_vector"].values, axis=0)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            emotion_map[str(news_id)] = np.concatenate([mean, std], axis=0).astype(np.float32)
        return emotion_map

    @staticmethod
    def _load_image_emotions(image_emotions_csv: str) -> dict[str, np.ndarray]:
        """Load image emotion probabilities by sample id."""
        df = pd.read_csv(image_emotions_csv)
        if "id" not in df.columns:
            raise ValueError("image_emotions_csv must contain an 'id' column.")

        return {
            str(row["id"]): np.array([row.get(key, 0.0) for key in IMAGE_EMOTION_KEYS], dtype=np.float32)
            for _, row in df.iterrows()
        }

    @staticmethod
    def _load_text_emotions(text_emotions_csv: str) -> dict[str, np.ndarray]:
        """Load text emotion probabilities by sample id."""
        df = pd.read_csv(text_emotions_csv)
        if "id" not in df.columns or "emotion" not in df.columns:
            raise ValueError("text_emotions_csv must contain 'id' and 'emotion' columns.")

        return {
            str(row["id"]): np.array(
                [float(parse_json_object(row["emotion"]).get(key, 0.0)) for key in TEXT_EMOTION_KEYS],
                dtype=np.float32,
            )
            for _, row in df.iterrows()
        }
