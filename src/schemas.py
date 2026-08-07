from dataclasses import dataclass

from pydantic import BaseModel, Field
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class PushbackResult(BaseModel):
    model_solution: str
    model_answer: float | None
    activations_path: str


class ExampleResult(BaseModel):
    question: str
    reference_solution: str
    reference_answer: float
    model_solution: str
    model_answer: float | None
    activations_path: str
    pushbacks: dict[str, PushbackResult] = Field(default_factory=dict)


class EvaluationRun(BaseModel):
    dataset_name: str
    model_name: str
    results: list[ExampleResult] = Field(default_factory=list)


@dataclass
class ModelBundle:
    """Wrapper for model and tokenizer to improve code readability."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
