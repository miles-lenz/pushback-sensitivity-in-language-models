import os

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Ensure HF token is loaded from .env file so we have access to the models.
load_dotenv()


SUPPORTED_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def load_model(model_alias: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face model and tokenizer."""

    # Input Validation
    if model_alias not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model alias: {model_alias}")

    # Token Validation
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    model_id = SUPPORTED_MODELS[model_alias]

    # The "auto" option automatically places the model on available GPUs.
    # If no GPU is found, we fall back to the CPU.
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    # Set up quantization configuration for 4-bit loading to save memory.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # Load the tokenizer and model separately. This provides more flexibility and allows
    # us later to extract activations from the model.
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map=device_map,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    return model, tokenizer
