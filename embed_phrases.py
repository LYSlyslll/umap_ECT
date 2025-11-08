"""Utility script to create concatenated text embeddings from JSONL inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate embeddings by concatenating Sentence-BERT embeddings "
            "of nested 'phrase' and 'keywords' texts within each JSONL record."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help=(
            "Path to the input JSONL file. Each line must contain a 'phrase' field "
            "whose value is a JSON string with nested 'phrase' and 'keywords' entries."
        ),
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output JSONL file that will contain 'phrase' and concatenated embeddings.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-BERT model name from Hugging Face Hub (default: all-mpnet-base-v2, 768 dimensions).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding texts (default: 32).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run the model on (e.g., 'cpu', 'cuda'). Defaults to SentenceTransformer's auto-detection.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Tuple[str, str]]:
    """Read nested JSON strings and extract phrase and keyword texts."""

    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            phrase_raw = payload.get("phrase")
            if not isinstance(phrase_raw, str):
                raise ValueError(
                    f"Line {line_number} must contain a string 'phrase' field with nested JSON."
                )

            try:
                nested_payload = json.loads(phrase_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number} contains invalid nested JSON in the 'phrase' field: {exc}"
                ) from exc

            nested_phrase = nested_payload.get("phrase")
            nested_keywords = nested_payload.get("keywords")

            if not isinstance(nested_phrase, str):
                raise ValueError(
                    f"Nested payload on line {line_number} must contain string 'phrase'."
                )

            if (
                not isinstance(nested_keywords, list)
                or len(nested_keywords) != 1
                or not isinstance(nested_keywords[0], str)
            ):
                raise ValueError(
                    f"Nested payload on line {line_number} must contain 'keywords' as a list "
                    "with a single string element."
                )

            records.append((nested_phrase, nested_keywords[0]))
    return records


def encode_texts(
    model: SentenceTransformer, texts: Iterable[str], batch_size: int
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def write_jsonl(path: Path, phrases: Iterable[str], embeddings: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        for phrase, embedding in zip(phrases, embeddings):
            record = {
                "phrase": phrase,
                "embedding": embedding.tolist(),
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    records = read_jsonl(args.input_file)
    if not records:
        raise ValueError("Input file does not contain any valid records.")

    phrases, keywords = zip(*records)

    model = SentenceTransformer(args.model_name, device=args.device)

    phrase_embeddings = encode_texts(model, phrases, batch_size=args.batch_size)
    keyword_embeddings = encode_texts(model, keywords, batch_size=args.batch_size)

    if phrase_embeddings.shape != keyword_embeddings.shape:
        raise RuntimeError("Mismatched embedding shapes between phrases and keywords.")

    concatenated_embeddings = np.concatenate(
        [phrase_embeddings, keyword_embeddings], axis=1
    )

    write_jsonl(args.output_file, phrases, concatenated_embeddings)


if __name__ == "__main__":
    main()
