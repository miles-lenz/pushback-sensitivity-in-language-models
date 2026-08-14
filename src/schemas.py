from dataclasses import dataclass

from pydantic import BaseModel, Field
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class PushbackResult(BaseModel):
    prompt_name: str
    model_solution: str
    model_answer: float | None
    activations_path: str


class ExampleResult(BaseModel):
    example_id: str
    question: str
    reference_solution: str
    reference_answer: float
    model_solution: str
    model_answer: float | None
    activations_path: str
    pushbacks: dict[str, PushbackResult] = Field(default_factory=dict)


@dataclass
class ModelBundle:
    """Wrapper for model and tokenizer to improve code readability."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
