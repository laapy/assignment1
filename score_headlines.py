import argparse
import sys
from datetime import datetime
from pathlib import Path

##cd "C:/Users/14277/Desktop/python for machine learning/assignment1/assignment"
##D:\python\python.exe "c:/Users/14277/Desktop/python for machine learning/assignment1/assignment/score_headlines.py" headlines_nyt_2024-12-02.txt nyt
##D:\python\python.exe "c:/Users/14277/Desktop/python for machine learning/assignment1/assignment/score_headlines.py" headlines_chicagotribune_2024-12-01.txt chicagotribune



import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"


print("PYTHON:", sys.executable)
print("NO_TF:", os.environ.get("TRANSFORMERS_NO_TF"))
print("NO_FLAX:", os.environ.get("TRANSFORMERS_NO_FLAX"))


import joblib
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Required:
      python score_headlines.py <input_file> <source>
    """
    parser = argparse.ArgumentParser(
        prog="score_headlines.py",
        description="Score news headlines as Optimistic/Pessimistic/Neutral using a pre-trained SVM model.",
        add_help=True,
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to a text file containing one headline per line",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Headline source string (e.g., nyt, chicagotribune)",
    )

    args = parser.parse_args()

    # Friendly error if missing required params (assignment requirement)
    if args.input_file is None or args.source is None:
        print("Usage: python score_headlines.py <input_file> <source>", file=sys.stderr)
        sys.exit(1)

    return args


def read_headlines(input_path: Path) -> list[str]:
    """
    Read headlines from a text file, one per line.
    Empty lines are skipped.
    """
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with input_path.open("r", encoding="utf-8") as f:
            headlines = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        # Fallback if file is not UTF-8 encoded
        with input_path.open("r", encoding="latin-1") as f:
            headlines = [line.strip() for line in f if line.strip()]

    return headlines


def load_embedder() -> SentenceTransformer:
    """
    Load the SentenceTransformer model.

    Prefer the local path on the linux server to avoid a large download:
      /opt/huggingface_models/all-MiniLM-L6-v2
    Fallback to model name which may download if not cached.
    """
    local_model_path = "/opt/huggingface_models/all-MiniLM-L6-v2"
    try:
        return SentenceTransformer(local_model_path)
    except Exception:
        return SentenceTransformer("all-MiniLM-L6-v2")


def load_svm_model(model_path: Path):
    """
    Load the pre-trained SVM model from disk.
    """
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error: failed to load model from {model_path}: {e}", file=sys.stderr)
        sys.exit(1)


def build_output_path(source: str) -> Path:
    """
    Output file name must be:
      headline_scores_<source>_<year>_<month>_<day>.txt
    Example:
      headline_scores_nyt_2025_01_15.txt
    """
    today = datetime.today()
    date_str = today.strftime("%Y_%m_%d")
    return Path(f"headline_scores_{source}_{date_str}.txt")


def write_scored_headlines(output_path: Path, labels, headlines) -> None:
    """
    Write results, one per line:
      <label>,<original headline>
    """
    with output_path.open("w", encoding="utf-8") as f:
        for label, headline in zip(labels, headlines):
            f.write(f"{label},{headline}\n")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    source = args.source.strip()

    headlines = read_headlines(input_path)
    if not headlines:
        print("Warning: input file contains no headlines (after removing empty lines).", file=sys.stderr)
        sys.exit(0)

    embedder = load_embedder()
    embeddings = embedder.encode(headlines)

    svm_model = load_svm_model(Path("svm.joblib"))
    labels = svm_model.predict(embeddings)

    if len(labels) != len(headlines):
        print(
            f"Error: number of predictions ({len(labels)}) does not match number of headlines ({len(headlines)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = build_output_path(source)
    write_scored_headlines(output_path, labels, headlines)

    print(f"Scored {len(headlines)} headlines.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
