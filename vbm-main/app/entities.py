from pydantic import BaseModel, ValidationError, UUID4, field_validator
from typing import Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    version: str

# API модели
class TaskResponse(BaseModel):
    task_id: UUID4
    message: str


class ProcessingTaskResponse(BaseModel):
    task_id: str
    type: str
    accounting_db: str
    status: str    


class StatusResponse(BaseModel):
    status: str
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    description: Optional[str] = None


class FittingIndicator(BaseModel):
    name: str
    id: str
    kind: str
    outer: bool
    numbers: list[str]
    analytics: Optional[list[str]] = None
    period_shifts: Optional[list[int]] = None
    period_year_numbers: Optional[list[int]] = None


class FittingParameters(BaseModel):
    model_id: str
    dimensions: list[str]
    indicators: list[FittingIndicator]
    data_filter: Optional[dict] = None


class TaskData(BaseModel):
    task_id: str
    type: str = 'FIT'
    status: str = "CREATED"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    accounting_db: Optional[str] = None
    replace: Optional[bool] = False
    model_id: Optional[str] = None
    fitting_parameters: Optional[FittingParameters] = None

    # Внутренние поля
    file_path: Optional[str] = None


class RawqDataStr(BaseModel, extra='allow'):
    period: str


class ColumnDescription(BaseModel):
    name: str
    analytics: dict
    analytic_key: str
    period_shift: int


class ModelInfo(BaseModel):
    model_id: str
    columns_descriptions: list[ColumnDescription]
