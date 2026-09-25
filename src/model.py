import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from schemas import ModelBundle

# Ensure HF token is loaded from .env file so we have access to the models.
load_dotenv()

logger = logging.getLogger("pipeline")


MODEL_CONFIG = {
    "do_sample": os.getenv("DO_SAMPLE", "0") == "1",
    "temperature": float(os.getenv("TEMPERATURE", "1")),
    "top_p": float(os.getenv("TOP_P", "1")),
    "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1")),
}


def load_model() -> tuple[ModelBundle, str]:
    """Load the Llama-3.2-3B model and tokenizer."""

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Require a Hugging Face token so model access is authorized.
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    # Get directory for cache from .env file and raise an
    # error if the path is invalid.
    cache_dir = os.getenv("HF_CACHE")
    if cache_dir is not None and not Path(cache_dir).exists():
        raise FileNotFoundError(f"[ERROR] Cache directory '{cache_dir}' is invalid.")

    use_quantization = os.getenv("USE_QUANTIZATION", "1") == "1"
    logger.info(f"Using quantization: {use_quantization}")

    # Use GPU when available and fall back to CPU otherwise.
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    # Enable 4-bit loading to reduce memory use during inference.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, cache_dir=cache_dir
    )

    # Ensure to pad on the left so the model generates
    # at the very end of the sequence.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map=device_map,
        dtype=torch.bfloat16,
        quantization_config=quantization_config if use_quantization else None,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )
    logger.info(f"Model config: {MODEL_CONFIG}")

    return ModelBundle(model=model, tokenizer=tokenizer), model_id


def get_model_response(
    model_bundle: ModelBundle, batch_messages: list[list[dict]]
) -> tuple[list[str], tuple]:
    """Generate a response for a batch and return their hidden states."""

    model, tokenizer = model_bundle.model, model_bundle.tokenizer

    # Format the chat history into model inputs and move them to the active device.
    inputs = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Extract Llama 3's specific terminator tokens
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Run generation without tracking gradients to save memory.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            **MODEL_CONFIG,
        )

    # Slice off the prompt tokens. Since we padded on the left, the new
    # tokens start at the exact same index for every sequence in the batch.
    input_length = inputs["input_ids"].shape[1]
    response_tokens = outputs.sequences[:, input_length:]

    # Decode all responses at once using batch_decode.
    response_texts = tokenizer.batch_decode(
        response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return response_texts, outputs.hidden_states


if __name__ == "__main__":
    load_model()
