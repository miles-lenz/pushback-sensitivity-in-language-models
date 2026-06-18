from dotenv import load_dotenv
from transformers import TextGenerationPipeline, pipeline

# Ensure HF token is loaded from .env file.
load_dotenv()


SUPPORTED_MODELS = {
    "llama": "meta-llama/Llama-3.2-3B",
}


def load_model(model_alias: str) -> TextGenerationPipeline:
    """Load a Hugging Face model."""

    # Input Validation
    if model_alias not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model alias: {model_alias}")

    model_id = SUPPORTED_MODELS[model_alias]
    return pipeline("text-generation", model=model_id)
