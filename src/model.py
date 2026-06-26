import os

import torch
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig, TextGenerationPipeline, pipeline

# Ensure HF token is loaded from .env file.
load_dotenv()


SUPPORTED_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def load_model(model_alias: str) -> TextGenerationPipeline:
    """Load a Hugging Face model."""

    # Input Validation
    if model_alias not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model alias: {model_alias}")

    # Token Validation
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    model_id = SUPPORTED_MODELS[model_alias]

    # In Hugging Face pipelines, device=0 means the first GPU and
    # device=-1 means CPU. We dynamically assign this.
    device_id = 0 if torch.cuda.is_available() else -1

    # Set up quantization configuration for 4-bit loading to save memory.
    model_kwargs = {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    }

    return pipeline(
        "text-generation",
        model=model_id,
        token=token,
        device=device_id,
        dtype=torch.bfloat16,
        model_kwargs=model_kwargs,
    )
