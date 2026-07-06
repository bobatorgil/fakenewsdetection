"""Extract BERTweet emotion scores for news text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

try:
    from .common import BertweetEmotionAnalyzer, append_csv, load_existing_ids, map_binary_label, read_json_or_jsonl
except ImportError:  # Allows: python preprocess/emotion.py
    from common import BertweetEmotionAnalyzer, append_csv, load_existing_ids, map_binary_label, read_json_or_jsonl


def resolve_id(record: dict[str, Any], id_field: str) -> Optional[str]:
    """Return the example id used by downstream feature files."""
    value = record.get(id_field)
    if value is None:
        return None
    return str(value).strip()


def resolve_label(record: dict[str, Any], label_field: Optional[str]) -> Optional[int]:
    """Return a binary label when a label field is provided."""
    if not label_field:
        return None
    return map_binary_label(record.get(label_field))


def build_record(
    example_id: str,
    emotion: dict[str, float],
    label: Optional[int] = None,
) -> dict[str, Any]:
    """Create a serializable emotion-feature record."""
    record: dict[str, Any] = {
        "id": example_id,
        "emotion": json.dumps(emotion, ensure_ascii=False),
    }
    if label is not None:
        record["label"] = int(label)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BERTweet emotion scores for news text.")
    parser.add_argument("--input_path", required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--output_csv", required=True, help="Output CSV file.")
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional output JSONL file. If omitted, it is derived from output_csv.",
    )
    parser.add_argument(
        "--input_format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Input file format. Use auto to infer from the file extension.",
    )
    parser.add_argument(
        "--id_field",
        default="id",
        help="Field used as the example id. For MMFakeBench, use --id_field image_path.",
    )
    parser.add_argument("--text_field", default="text", help="Field containing the news text.")
    parser.add_argument(
        "--label_field",
        default=None,
        help="Optional label field. Common values such as fake/true/0/1 are mapped to 0=fake and 1=real.",
    )
    parser.add_argument("--model_name", default="finiteautomata/bertweet-base-emotion-analysis")
    parser.add_argument("--max_text_chars", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in output_csv.")
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

    done_ids = load_existing_ids(args.output_csv, "id") if args.resume else set()
    if done_ids:
        print(f"Resume: found {len(done_ids)} processed examples in {args.output_csv}")

    records = read_json_or_jsonl(args.input_path, args.input_format)
    buffer: list[dict[str, Any]] = []
    total_written = 0

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    jsonl_mode = "a" if args.resume else "w"
    with Path(output_jsonl).open(jsonl_mode, encoding="utf-8") as jsonl_file:
        for record in tqdm(records, desc="Extracting text emotions"):
            example_id = resolve_id(record, args.id_field)
            if not example_id or example_id in done_ids:
                continue

            text = str(record.get(args.text_field, "")).strip()
            if not text:
                continue

            label = resolve_label(record, args.label_field)
            emotion = analyzer.predict(text)
            output_record = build_record(example_id, emotion, label)

            buffer.append(output_record)
            jsonl_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            done_ids.add(example_id)

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
