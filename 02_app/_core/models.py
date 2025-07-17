from pydantic import BaseModel
from typing import List


class SearchQueries(BaseModel):
    queries: List[str]


class RelevanceCheck(BaseModel):
    reasoning: str
    relevance: bool | None


class ReflectTask(BaseModel):
    reflection: str
    finished: bool | None
