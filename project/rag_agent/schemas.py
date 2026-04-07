from typing import List
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    """
    Pydantic schema used for structured LLM outputs during the query rewrite phase.
    This ensures the model output is always a strictly valid JSON object.
    """
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )