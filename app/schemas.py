from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class MathAnalysisSchema(BaseModel):

    answer: str = Field(..., description="Final answer or choice (A, B, C, D)")
    topic: List[str] = Field(..., description="Corresponding topics of SAT Math Taxonomy")
    sub_topic: List[str] = Field(..., desciption="Corresponding sub-topics of SAT Math Taxonomy")

    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="difficulty")
    question_type: Literal["multiple_choice", "free_response"] = Field(..., description="Question Type")

    image_path: Optional[str] = None

class AnalysisRequest(BaseModel):

    image_path: str
    custom_prompt: Optional[str] = None
