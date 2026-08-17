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
    cosine_similarity: float | None = None


class ExampleResult(BaseModel):
    example_id: str
    question: str
    reference_solution: str
    reference_answer: float
    model_solution: str
    model_answer: float | None
    activations_path: str
    pushbacks: dict[str, PushbackResult] = Field(default_factory=dict)


class EvaluationRun(BaseModel):
    run_id: str
    dataset_name: str
    model_name: str
    results: list[ExampleResult] = Field(default_factory=list)


@dataclass
class ModelBundle:
    """Wrapper for model and tokenizer to improve code readability."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
