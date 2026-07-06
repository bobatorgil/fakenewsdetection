"""Generate Reddit-style social reactions for news examples using an Ollama model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm import tqdm

try:
    from .common import read_jsonl, write_jsonl
except ImportError:  # Allows: python preprocess/comment_generator.py
    from common import read_jsonl, write_jsonl

PROMPT_TEMPLATE = """You are simulating realistic Reddit users commenting on a news post.

Your role is ONLY to simulate users and write comments.
Do NOT analyze emotions, do NOT label stance, do NOT explain anything.

PERSONA OPTIONS (choose only from these):
- Gender: [male, female]
- Age Group: [under 17; 18 to 29; 30 to 49; 50 to 64; over 65]
- Education: [college graduate; some college (no degree); high school or less]

TASK:
1) For this specific news, choose 5 DIFFERENT personas who would realistically be interested.
2) Write exactly ONE Reddit-style comment per persona.

INPUT:
[NEWS_TEXT]
{news_text}

OUTPUT RULES:
- Output must be a valid JSON array with exactly 5 objects.
- Each object must contain:
  - persona: (gender, age_group, education)
  - comment: a Reddit-style comment (1-3 sentences)
- Personas must all be different (no duplicate combinations).
- Use at least 3 different age groups and at least 2 different education levels.
- Do NOT mention persona attributes in the comment.
- Comments must sound human and Reddit-like: informal, imperfect, sometimes sarcastic.
- Each comment must be 20 words or fewer.
- Emotional reactions should be expressed implicitly through wording, tone, punctuation, slang, or style.
- At least 3 of the 5 comments should clearly convey emotional reactions through text.
- Do NOT be overly formal or essay-like.
- No moral lecturing, no summarizing the news.
- No hate speech, no harassment, no graphic violence.
- No meta talk about being an AI or generating comments.
"""


def find_image_path(images_dir: str | os.PathLike[str], example_id: str) -> Optional[Path]:
    """Find a JPG or PNG image for an example id."""
    images_dir = Path(images_dir)
    for extension in (".jpg", ".png", ".jpeg", ".webp"):
        candidate = images_dir / f"{example_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def encode_image_base64(image_path: str | os.PathLike[str]) -> str:
    """Encode an image file for Ollama multimodal requests."""
    with Path(image_path).open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_array(text: str) -> Optional[list[dict[str, Any]]]:
    """Parse a JSON array from a raw model response."""
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else None
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, list) else None
    except json.JSONDecodeError:
        return None


def normalize_persona(value: Any) -> Optional[dict[str, str]]:
    """Normalize persona output to a dictionary."""
    if isinstance(value, dict):
        return {
            "gender": str(value.get("gender", "")).strip(),
            "age_group": str(value.get("age_group", "")).strip(),
            "education": str(value.get("education", "")).strip(),
        }

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) == 3:
            return {"gender": parts[0], "age_group": parts[1], "education": parts[2]}

    if isinstance(value, list) and len(value) == 3:
        return {
            "gender": str(value[0]).strip(),
            "age_group": str(value[1]).strip(),
            "education": str(value[2]).strip(),
        }

    return None


def stable_seed_from_id(example_id: str) -> int:
    """Create a deterministic seed that is stable across Python processes."""
    digest = hashlib.md5(example_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def validate_comments(items: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    """Validate and normalize generated comment objects."""
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} comments, but received {len(items)}.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Comment item {index} is not a JSON object.")

        persona = normalize_persona(item.get("persona"))
        comment = item.get("comment")
        if persona is None:
            raise ValueError(f"Comment item {index} has an invalid persona: {item.get('persona')}")
        if not isinstance(comment, str) or not comment.strip():
            raise ValueError(f"Comment item {index} has an invalid comment.")

        normalized.append({"persona": persona, "comment": comment.strip()})
    return normalized


def request_ollama_comments(
    ollama_url: str,
    model: str,
    news_text: str,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    num_predict: int,
    expected_count: int,
    image_b64: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Request comments from an Ollama generation endpoint."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": PROMPT_TEMPLATE.format(news_text=news_text),
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)
    if image_b64 is not None:
        payload["images"] = [image_b64]

    response = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=600)
    response.raise_for_status()

    raw_response = response.json().get("response", "")
    parsed = parse_json_array(raw_response)
    if parsed is None:
        raise ValueError(f"Failed to parse a JSON array from the model response: {raw_response[:500]}")
    return validate_comments(parsed, expected_count)


def load_completed_ids(output_path: str | os.PathLike[str]) -> set[str]:
    """Load ids that already have generated output."""
    path = Path(output_path)
    if not path.exists():
        return set()

    completed: set[str] = set()
    for record in read_jsonl(path):
        news_id = record.get("news_id")
        if news_id is not None and not record.get("error"):
            completed.add(str(news_id))
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Reddit-style comments for news examples.")
    parser.add_argument("--data_dir", required=True, help="Directory containing split JSONL files.")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--input_suffix", default=".jsonl")
    parser.add_argument("--output_suffix", default="_llm_comments.jsonl")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--use_images", action="store_true", help="Attach images to Ollama requests when available.")

    parser.add_argument("--ollama_url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5vl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_predict", type=int, default=256)
    parser.add_argument("--num_comments", type=int, default=5)

    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum samples per split. Use -1 for all samples.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between requests.")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the output file.")
    parser.add_argument("--seed_mode", choices=["none", "fixed", "by_id"], default="by_id")
    parser.add_argument("--fixed_seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.data_dir) / args.images_subdir

    for split in args.splits:
        input_path = Path(args.data_dir) / f"{split}{args.input_suffix}"
        output_path = Path(args.data_dir) / f"{split}{args.output_suffix}"

        rows = list(read_jsonl(input_path))
        rows = rows[max(0, args.start_idx) :]
        if args.max_samples != -1:
            rows = rows[: args.max_samples]

        output_records = list(read_jsonl(output_path)) if args.resume and output_path.exists() else []
        completed_ids = load_completed_ids(output_path) if args.resume else set()

        progress = tqdm(rows, desc=f"Generating comments for {split}")
        for row in progress:
            example_id = str(row.get(args.id_field, "")).strip()
            if not example_id or example_id in completed_ids:
                continue

            news_text = str(row.get(args.text_field, "")).strip()
            if not news_text:
                continue

            seed: Optional[int]
            if args.seed_mode == "none":
                seed = None
            elif args.seed_mode == "fixed":
                seed = args.fixed_seed
            else:
                seed = stable_seed_from_id(example_id)

            image_b64 = None
            if args.use_images:
                image_path = find_image_path(images_dir, example_id)
                if image_path is not None:
                    image_b64 = encode_image_base64(image_path)

            try:
                comments = request_ollama_comments(
                    ollama_url=args.ollama_url,
                    model=args.model,
                    news_text=news_text,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                    num_predict=args.num_predict,
                    expected_count=args.num_comments,
                    image_b64=image_b64,
                )
                for comment in comments:
                    comment["news_id"] = example_id

                output_records.append({"news_id": example_id, "comments": comments})
                completed_ids.add(example_id)
            except Exception as error:
                output_records.append({"news_id": example_id, "error": str(error)})

            if len(output_records) % args.save_every == 0:
                write_jsonl(output_records, output_path)

            if args.sleep > 0:
                time.sleep(args.sleep)

        write_jsonl(output_records, output_path)
        print(f"Done: {split} -> {output_path} ({len(output_records)} rows)")


if __name__ == "__main__":
    main()
