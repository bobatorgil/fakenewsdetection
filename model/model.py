"""Neural network modules for multimodal fake news detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

from .dataset import IMAGE_EMOTION_KEYS, TEXT_EMOTION_KEYS

TEXT_MODEL_NAME = "bert-base-uncased"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224-in21k"


class DualEmotionFusion(nn.Module):
    """Cross-modal attention module for text and image emotion distributions."""

    def __init__(self, text_dim: int = 7, image_dim: int = 9, hidden_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.text_to_image_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.image_to_text_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_emotion: torch.Tensor, image_emotion: torch.Tensor) -> torch.Tensor:
        text_feature = self.text_projection(self.text_encoder(text_emotion)).unsqueeze(1)
        image_feature = self.image_projection(self.image_encoder(image_emotion)).unsqueeze(1)

        text_attended, _ = self.text_to_image_attention(text_feature, image_feature, image_feature)
        image_attended, _ = self.image_to_text_attention(image_feature, text_feature, text_feature)

        text_attended = self.text_norm(text_attended + text_feature)
        image_attended = self.image_norm(image_attended + image_feature)
        return torch.cat([text_attended.squeeze(1), image_attended.squeeze(1)], dim=-1)


class MultimodalGatedFakeNewsModel(nn.Module):
    """BERT-ViT model with gated comment and emotion auxiliary branches.

    The base branch uses the news text and image representations. The auxiliary
    branch uses comment text, fused publisher-side emotions, and comment emotion
    statistics. The auxiliary branch is gated and added to the base branch through
    residual fusion before binary classification.
    """

    def __init__(self, projection_dim: int = 768, emotion_projection_dim: int = 384, dropout: float = 0.5) -> None:
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(TEXT_MODEL_NAME)
        self.image_encoder = ViTModel.from_pretrained(IMAGE_MODEL_NAME)

        self.dual_emotion_fusion = DualEmotionFusion(
            text_dim=len(TEXT_EMOTION_KEYS),
            image_dim=len(IMAGE_EMOTION_KEYS),
            hidden_dim=128,
            num_heads=4,
        )
        self.comment_emotion_encoder = nn.Sequential(
            nn.Linear(len(TEXT_EMOTION_KEYS) * 2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.text_projection = self._make_projection(768, projection_dim, dropout)
        self.image_projection = self._make_projection(768, projection_dim, dropout)
        self.comment_projection = self._make_projection(768, projection_dim, dropout)
        self.publisher_emotion_projection = self._make_projection(256, emotion_projection_dim, dropout)
        self.comment_emotion_projection = self._make_projection(256, emotion_projection_dim, dropout)

        self.comment_gate = nn.Sequential(nn.Linear(projection_dim, 1), nn.Sigmoid())
        self.publisher_emotion_gate = nn.Sequential(nn.Linear(emotion_projection_dim, 1), nn.Sigmoid())
        self.comment_emotion_gate = nn.Sequential(nn.Linear(emotion_projection_dim, 1), nn.Sigmoid())
        self.auxiliary_scale = nn.Parameter(torch.tensor(1.0))

        fused_dim = projection_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

    @staticmethod
    def _make_projection(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        input_ids_text: torch.Tensor,
        attention_text: torch.Tensor,
        pixel_values: torch.Tensor,
        emotion_text: torch.Tensor,
        emotion_image: torch.Tensor,
        emotion_comment_stats: torch.Tensor,
        input_ids_comment: torch.Tensor,
        attention_comment: torch.Tensor,
    ) -> torch.Tensor:
        text_outputs = self.text_encoder(input_ids=input_ids_text, attention_mask=attention_text)
        text_vector = text_outputs.last_hidden_state[:, 0, :]

        comment_outputs = self.text_encoder(input_ids=input_ids_comment, attention_mask=attention_comment)
        comment_vector = comment_outputs.last_hidden_state[:, 0, :]

        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_vector = image_outputs.pooler_output
        if image_vector is None:
            image_vector = image_outputs.last_hidden_state[:, 0, :]

        publisher_emotion = self.dual_emotion_fusion(emotion_text, emotion_image)
        comment_emotion = self.comment_emotion_encoder(emotion_comment_stats)

        text_vector = self.text_projection(text_vector)
        image_vector = self.image_projection(image_vector)
        comment_vector = self.comment_projection(comment_vector)
        publisher_emotion = self.publisher_emotion_projection(publisher_emotion)
        comment_emotion = self.comment_emotion_projection(comment_emotion)

        base_feature = torch.cat([text_vector, image_vector], dim=1)

        gated_comment = self.comment_gate(comment_vector) * comment_vector
        gated_publisher_emotion = self.publisher_emotion_gate(publisher_emotion) * publisher_emotion
        gated_comment_emotion = self.comment_emotion_gate(comment_emotion) * comment_emotion

        auxiliary_feature = torch.cat(
            [gated_comment, gated_publisher_emotion, gated_comment_emotion],
            dim=1,
        )
        fused_feature = base_feature + self.auxiliary_scale * auxiliary_feature
        return self.classifier(fused_feature)
