"""Training, validation, and test routines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .utils import save_json

FAKE_LABEL = 0


@dataclass
class ClassificationMetrics:
    """Container for binary classification metrics."""

    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> ClassificationMetrics:
    """Compute accuracy and fake-class precision, recall, and F1-score."""
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision_fake=precision_score(y_true, y_pred, pos_label=FAKE_LABEL, zero_division=0),
        recall_fake=recall_score(y_true, y_pred, pos_label=FAKE_LABEL, zero_division=0),
        f1_fake=f1_score(y_true, y_pred, pos_label=FAKE_LABEL, zero_division=0),
    )


def set_backbone_trainable(model: nn.Module, train_backbone: bool) -> None:
    """Enable or disable gradient updates for the BERT and ViT backbones."""
    backbone_prefixes = ("text_encoder.", "image_encoder.")
    for name, parameter in model.named_parameters():
        if name.startswith(backbone_prefixes):
            parameter.requires_grad = train_backbone
        else:
            parameter.requires_grad = True


def build_optimizer(
    model: nn.Module,
    learning_rate_head: float,
    learning_rate_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build AdamW with separate learning rates for backbone and task-specific layers."""
    backbone_prefixes = ("text_encoder.", "image_encoder.")
    backbone_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(backbone_prefixes):
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": learning_rate_backbone})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": learning_rate_head})

    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensors in a batch to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run a forward pass using the standardized batch dictionary."""
    return model(
        input_ids_text=batch["input_ids_text"],
        attention_text=batch["attention_text"],
        pixel_values=batch["pixel_values"],
        emotion_text=batch["emotion_text"],
        emotion_image=batch["emotion_image"],
        emotion_comment_stats=batch["emotion_comment_stats"],
        input_ids_comment=batch["input_ids_comment"],
        attention_comment=batch["attention_comment"],
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    """Train the model for one epoch and return the mean loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = forward_batch(model, batch)
            loss = criterion(logits, batch["label"])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, ClassificationMetrics]:
    """Evaluate the model and return mean loss plus classification metrics."""
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = forward_batch(model, batch)
            loss = criterion(logits, batch["label"])

        total_loss += loss.item()
        predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())

    mean_loss = total_loss / max(1, len(data_loader))
    return mean_loss, compute_metrics(labels, predictions)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    args: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Run the full training procedure and evaluate the best validation checkpoint."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()
    best_validation_accuracy = 0.0
    best_model_path = output_dir / "best_model.pt"

    history: dict[str, list[float | int]] = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_precision_fake": [],
        "validation_recall_fake": [],
        "validation_f1_fake": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_backbone = epoch <= args.backbone_warmup_epochs
        set_backbone_trainable(model, train_backbone)
        phase = "backbone tuning" if train_backbone else "head training"
        print(f">> Epoch {epoch}: {phase}")

        optimizer = build_optimizer(
            model=model,
            learning_rate_head=args.lr_head,
            learning_rate_backbone=args.lr_backbone,
            weight_decay=args.weight_decay,
        )

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
        )
        validation_loss, validation_metrics = evaluate(model, validation_loader, criterion, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"accuracy={validation_metrics.accuracy:.4f} "
            f"precision_fake={validation_metrics.precision_fake:.4f} "
            f"recall_fake={validation_metrics.recall_fake:.4f} "
            f"f1_fake={validation_metrics.f1_fake:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["validation_loss"].append(round(validation_loss, 4))
        history["validation_accuracy"].append(round(validation_metrics.accuracy, 4))
        history["validation_precision_fake"].append(round(validation_metrics.precision_fake, 4))
        history["validation_recall_fake"].append(round(validation_metrics.recall_fake, 4))
        history["validation_f1_fake"].append(round(validation_metrics.f1_fake, 4))
        save_json(history, output_dir / "train_history.json")

        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    summary = {
        "best_validation_accuracy": float(best_validation_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics.accuracy),
        "test_precision_fake": float(test_metrics.precision_fake),
        "test_recall_fake": float(test_metrics.recall_fake),
        "test_f1_fake": float(test_metrics.f1_fake),
        "positive_label": FAKE_LABEL,
        "epochs": int(args.epochs),
        "backbone_warmup_epochs": int(args.backbone_warmup_epochs),
        "lr_head": float(args.lr_head),
        "lr_backbone": float(args.lr_backbone),
        "weight_decay": float(args.weight_decay),
    }
    save_json(summary, output_dir / "summary.json")

    print(
        f"[TEST] accuracy={test_metrics.accuracy:.4f} "
        f"precision_fake={test_metrics.precision_fake:.4f} "
        f"recall_fake={test_metrics.recall_fake:.4f} "
        f"f1_fake={test_metrics.f1_fake:.4f}"
    )
    return summary
