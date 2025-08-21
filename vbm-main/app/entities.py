from pydantic import BaseModel, ValidationError, UUID4, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum


class ModelStatuses(Enum):
    CREATED = 'CREATED'
    FITTING = 'FITTING'    
    READY = 'READY'
    ERROR = 'ERROR'


class FIStatuses(Enum):
    NOT_CALCULATED = 'NOT_CALCULATED'
    CALCULATING = 'CALCULATING'    
    CALCULATED = 'CALCULATED'
    ERROR = 'ERROR'    


class ModelTypes(Enum):
    pf = 'pf'
    nn = 'nn'    


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


class FittingParameters(BaseModel, extra='allow'):
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
    replace: Optional[bool] = False
    model_id: Optional[str] = None
    model_type: Optional[ModelTypes] = None
    fitting_parameters: Optional[FittingParameters] = None

    # Внутренние поля
    file_path: Optional[str] = None


class RawDataStr(BaseModel, extra='allow'):
    period: str
    accounting_db: str


class ColumnDescription(BaseModel):
    name: str
    analytics: dict
    analytic_key: str
    period_shift: int


class ModelInfo(BaseModel):
    model_id: str
    columns_descriptions: list[ColumnDescription]
    status: ModelStatuses
    fi_status: FIStatuses
    error_text: str
    fi_error_text: str   


class FeatureImportances(BaseModel):
    fi: dict[str, float]
    descr: dict[str, dict] 


class SAInputData(BaseModel):
    data: list[RawDataStr]
    deviations: list[float]
    input_indicators: list[str]
    output_indicator: str
    get_graph: bool
    auto_selection_number: int


class SARow(BaseModel):
    ind_id: str
    ind_kind: str
    coef: float
    value: float
    value_0: float
    delta: float
    relative_delta: float


class SAOutputData(BaseModel):
    data: list[SARow]
    graph: str


class FAScenario(BaseModel):
    id: str
    name: str
    value: float


class FAScenarios(BaseModel):
    base: FAScenario
    calculated: FAScenario


class FAInputData(BaseModel):
    data: list[RawDataStr]
    scenarios: FAScenarios
    input_indicators: list[str]
    output_indicator: str
    get_graph: bool


class FARow(BaseModel):
    ind_id: str
    ind_kind: str
    value: float


class FAOutputData(BaseModel):
    data: list[FARow]
    graph: str






