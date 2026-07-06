"""Extract BERTweet emotion scores for generated or real comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

try:
    from .common import BertweetEmotionAnalyzer, append_csv, read_jsonl
except ImportError:  # Allows: python preprocess/comment_emotion.py
    from common import BertweetEmotionAnalyzer, append_csv, read_jsonl


def iter_comment_records(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield flattened comment records with news_id and comment fields.

    Supported input formats:
    1. {"news_id": "...", "comments": [{"comment": "..."}, ...]}
    2. {"news_id": "...", "comment": "..."}
    """
    for obj in read_jsonl(path):
        if obj.get("error"):
            continue

        news_id = str(obj.get("news_id", obj.get("id", ""))).strip()
        if not news_id:
            continue

        if isinstance(obj.get("comments"), list):
            for item in obj["comments"]:
                if isinstance(item, dict):
                    comment = str(item.get("comment", "")).strip()
                else:
                    comment = str(item).strip()
                if comment:
                    yield {"news_id": news_id, "comment": comment}
            continue

        comment = str(obj.get("comment", "")).strip()
        if comment:
            yield {"news_id": news_id, "comment": comment}


def build_record(news_id: str, comment: str, emotion: dict[str, float]) -> dict[str, Any]:
    """Create a serializable output record."""
    return {
        "news_id": news_id,
        "comment": comment,
        "emotion": json.dumps(emotion, ensure_ascii=False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BERTweet emotion scores for comments.")
    parser.add_argument(
        "--comments_jsonl",
        required=True,
        help="Input JSONL file containing generated or real comments.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV file. The emotion column stores a JSON-encoded emotion dictionary.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional output JSONL file. If omitted, it is derived from output_csv.",
    )
    parser.add_argument("--model_name", default="finiteautomata/bertweet-base-emotion-analysis")
    parser.add_argument("--max_text_chars", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--resume", action="store_true", help="Skip comments already present in output_csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = args.output_jsonl or str(Path(args.output_csv).with_suffix(".jsonl"))

    analyzer = BertweetEmotionAnalyzer(
        model_name=args.model_name,
        max_text_chars=args.max_text_chars,
        max_tokens=args.max_tokens,
    )
    print(f"Device: {analyzer.device}")

    done_keys: set[str] = set()
    if args.resume:
        # CSV resume is enough for this script because it contains both columns.
        import csv

        output_csv = Path(args.output_csv)
        if output_csv.exists():
            with output_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_keys.add(f"{row.get('news_id', '')}\t{row.get('comment', '')}")
            print(f"Resume: found {len(done_keys)} processed comments in {output_csv}")

    buffer: list[dict[str, Any]] = []
    total_written = 0

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    jsonl_mode = "a" if args.resume else "w"
    with Path(output_jsonl).open(jsonl_mode, encoding="utf-8") as jsonl_file:
        for item in tqdm(iter_comment_records(args.comments_jsonl), desc="Extracting comment emotions"):
            news_id = item["news_id"]
            comment = item["comment"]
            key = f"{news_id}\t{comment}"
            if key in done_keys:
                continue

            emotion = analyzer.predict(comment)
            record = build_record(news_id, comment, emotion)
            buffer.append(record)
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_keys.add(key)

            if len(buffer) >= args.save_every:
                append_csv(buffer, args.output_csv)
                total_written += len(buffer)
                print(f"Saved {total_written} rows to {args.output_csv}")
                buffer.clear()

    if buffer:
        append_csv(buffer, args.output_csv)
        total_written += len(buffer)

    print(f"Done. Newly written rows: {total_written}")
    print(f"CSV: {args.output_csv}")
    print(f"JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
