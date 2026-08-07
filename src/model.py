import os

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


SUPPORTED_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def load_model(model_alias: str) -> ModelBundle:
    """Load a supported model and tokenizer."""

    # Validate the requested alias before loading anything.
    if model_alias not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model alias: {model_alias}")

    # Require a Hugging Face token so model access is authorized.
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    model_id = SUPPORTED_MODELS[model_alias]

    # Use GPU when available and fall back to CPU otherwise.
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    # Enable 4-bit loading to reduce memory use during inference.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # Load the tokenizer and model separately so activations can be inspected later.
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map=device_map,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    return ModelBundle(model=model, tokenizer=tokenizer)


def get_model_response(
    model_bundle: ModelBundle, messages: list[dict]
) -> tuple[str, tuple]:
    """Generate a response and return its hidden states."""

    model, tokenizer = model_bundle.model, model_bundle.tokenizer

    # Format the chat history into model inputs and move them to the active device.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Run generation without tracking gradients to save memory.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            repetition_penalty=1.15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # Slice off the prompt tokens so only the newly generated text is decoded.
    response_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1] :]
    response_text = tokenizer.decode(
        response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return response_text, outputs.hidden_states
