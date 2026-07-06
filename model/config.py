"""Command-line configuration for the training script."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a multimodal fake news detection model with text, image, comment, and emotion features."
    )

    parser.add_argument(
        "--train_jsonl",
        default="/home/iai4/Desktop/ohw/data/fakeddit/real_comments_5_filtered_splits/train_with_emotion_real_comments_5.jsonl",
        help="Path to the training split in JSONL format.",
    )
    parser.add_argument(
        "--dev_jsonl",
        default="/home/iai4/Desktop/ohw/data/fakeddit/real_comments_5_filtered_splits/dev_with_emotion_real_comments_5.jsonl",
        help="Path to the validation split in JSONL format.",
    )
    parser.add_argument(
        "--test_jsonl",
        default="/home/iai4/Desktop/ohw/data/fakeddit/real_comments_5_filtered_splits/test_with_emotion_real_comments_5.jsonl",
        help="Path to the test split in JSONL format.",
    )
    parser.add_argument(
        "--data_dir",
        default="/home/iai4/Desktop/ohw/data/fakeddit",
        help="Root directory of the Fakeddit dataset. Images are expected under data_dir/images.",
    )
    parser.add_argument(
        "--image_emotions_csv",
        default="/home/iai4/Desktop/ohw/data/fakeddit/emotionclip_imagewise_results.csv",
        help="CSV file containing image emotion features.",
    )
    parser.add_argument(
        "--text_emotions_csv",
        default="/home/iai4/Desktop/ohw/data/fakeddit/fakeddit_bertweet_emotions.csv",
        help="CSV file containing text emotion features.",
    )
    parser.add_argument(
        "--comment_emotions_csv",
        default="/home/iai4/Desktop/ohw/data/fakeddit/llm_comments_bertweet_emotions.csv",
        help="CSV file containing comment emotion features.",
    )
    parser.add_argument(
        "--comments_jsonl",
        default="/home/iai4/Desktop/ohw/data/fakeddit/llm_comments.jsonl",
        help="JSONL file containing comments grouped by news_id or one comment per line.",
    )
    parser.add_argument("--label_column", default="2_way_label", help="Name of the binary label column.")

    parser.add_argument("--max_text_length", type=int, default=128, help="Maximum token length for news text.")
    parser.add_argument("--max_comment_length", type=int, default=256, help="Maximum token length for merged comments.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training and evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader worker processes.")

    parser.add_argument("--epochs", type=int, default=20, help="Total number of training epochs.")
    parser.add_argument(
        "--backbone_warmup_epochs",
        type=int,
        default=3,
        help="Number of initial epochs where BERT and ViT are trainable.",
    )
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for task-specific layers.")
    parser.add_argument("--lr_backbone", type=float, default=5e-6, help="Learning rate for BERT and ViT backbones.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability.")
    parser.add_argument("--projection_dim", type=int, default=768, help="Projection size for text, image, and comment branches.")
    parser.add_argument(
        "--emotion_projection_dim",
        type=int,
        default=384,
        help="Projection size for emotion branches.",
    )
    parser.add_argument("--output_dir", default="results_gated_fusion", help="Directory for checkpoints and logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser
