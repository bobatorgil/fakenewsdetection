"""Entry point for training the multimodal fake news detection model."""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor

from model.config import build_parser
from model.dataset import FakedditMultimodalDataset
from model.model import MultimodalGatedFakeNewsModel
from model.train import fit
from model.utils import get_device, set_seed

TEXT_MODEL_NAME = "bert-base-uncased"
IMAGE_MODEL_NAME = "google/vit-base-patch16-224-in21k"


def build_dataloaders(args):
    """Create train, validation, and test DataLoaders."""
    tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_processor = ViTImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    images_dir = os.path.join(args.data_dir, "images")

    dataset_kwargs = {
        "images_dir": images_dir,
        "image_emotions_csv": args.image_emotions_csv,
        "text_emotions_csv": args.text_emotions_csv,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "max_text_length": args.max_text_length,
        "comment_emotions_csv": args.comment_emotions_csv,
        "comments_jsonl": args.comments_jsonl,
        "max_comment_length": args.max_comment_length,
        "label_column": args.label_column,
    }

    train_dataset = FakedditMultimodalDataset(jsonl_path=args.train_jsonl, **dataset_kwargs)
    validation_dataset = FakedditMultimodalDataset(jsonl_path=args.dev_jsonl, **dataset_kwargs)
    test_dataset = FakedditMultimodalDataset(jsonl_path=args.test_jsonl, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, validation_loader, test_loader


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, validation_loader, test_loader = build_dataloaders(args)
    model = MultimodalGatedFakeNewsModel(
        projection_dim=args.projection_dim,
        emotion_projection_dim=args.emotion_projection_dim,
        dropout=args.dropout,
    ).to(device)

    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        args=args,
        device=device,
    )


if __name__ == "__main__":
    main()
