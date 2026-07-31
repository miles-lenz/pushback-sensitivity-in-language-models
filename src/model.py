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


def get_model_response(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, question: str) -> tuple[str, tuple]:
    """..."""

    messages = [
        # todo: add system prompt
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # ...
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # ...
    response_tokens = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(
        response_tokens, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return response_text, outputs.hidden_states
