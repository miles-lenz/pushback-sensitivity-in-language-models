import os
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

# Load the HF token into the system's environment so that the initial
# download of datasets can proceed without any rate limits.
load_dotenv()


def load_gsm8k_dataset() -> tuple[Dataset, str]:
    """Load the gsm8k test-dataset from Hugging Face."""

    # Get directory for cache from .env file and raise an
    # error if the path is invalid.
    cache_dir = os.getenv("HF_CACHE")
    if cache_dir is not None and not Path(cache_dir).exists():
        raise FileNotFoundError(f"[ERROR] Cache directory '{cache_dir}' is invalid.")

    # The dataset is automatically cached in the Hugging Face
    # cache directory, so it will only be downloaded once.
    dataset = load_dataset(
        path="openai/gsm8k",
        name="main",
        cache_dir=cache_dir,
    )

    return dataset["test"], "gsm8k"


if __name__ == "__main__":
    load_gsm8k_dataset()
