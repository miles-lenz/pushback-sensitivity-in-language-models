from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv

load_dotenv()


def load_gsm8k_dataset() -> Dataset:
    """Load the gsm8k dataset from Hugging Face."""

    # The dataset is automatically cached in the Hugging Face
    # cache directory, so it will only be downloaded once.
    dataset = load_dataset("openai/gsm8k", "main")

    # Since we don't need a train/test split, we can merge
    # them into a single unified dataset.
    splits = list(dataset.values())
    dataset = concatenate_datasets(splits)

    return dataset
