from pydantic import BaseModel, Field


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
