from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
	image_id: int | None = None
	image_ids: list[int] | None = None
	question: str = Field(min_length=1)
	requested_capability: str = "auto"


class AnalyzeRequest(BaseModel):
	image_id: int | None = None
	image_ids: list[int] | None = None
	question: str = Field(min_length=1)
	requested_capability: str = "auto"


class AnalysisCreateResponse(BaseModel):
	analysis_id: int
	status: str
	plan: dict[str, Any] | None = None


class ResultItemResponse(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	result_type: str | None
	data: dict[str, Any] | None
	created_at: datetime


class AnalysisResultResponse(BaseModel):
	analysis_id: int
	image_id: int | None
	question: str | None
	status: str
	results: list[ResultItemResponse]
	plan: dict[str, Any] | None = None
	final_result: dict[str, Any] | None = None
	trace: dict[str, Any] | None = None


class ExecutionResponse(BaseModel):
	id: int
	step: str
	specialist: str | None = None
	status: str
	started_at: datetime | None
	completed_at: datetime | None
	error_message: str | None
	created_at: datetime
	result_data: dict[str, Any] | None = None