import uuid
import logging
from fastapi import APIRouter, Header, HTTPException, BackgroundTasks, File, UploadFile, Depends, Query, Path, Body
from typing import Optional

import config
from entities import (HealthResponse, 
                      RawDataStr, 
                      TaskResponse, 
                      StatusResponse, 
                      FittingParameters, 
                      ProcessingTaskResponse, 
                      ModelInfo,
                      ModelTypes,
                      FeatureImportances, 
                      SAInputData,
                      SAOutputData,
                      SARow,
                      FAInputData,
                      FAOutputData,
                      FARow)

from storage import task_storage
from data_processing import data_loader
from models import Model, model_manager
from post_processing import post_processor

from cachetools import TTLCache
import aiohttp
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

logging.getLogger("vbm_api_logger").setLevel(logging.ERROR)

auth_cache = TTLCache(maxsize=1000, ttl=300)


async def check_token(token: str = Header()) -> bool:
    """Проверка токена авторизации с кэшем и защитой от сбоев."""
    if config.TEST_MODE: # type: ignore
        return True

    if token in auth_cache:
        return True

    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{config.AUTH_SERVICE_URL}/check-token", # type: ignore
                json={"token": token}
            ) as response:
                if response.status == 200:
                    auth_cache[token] = True
                    return True
                elif response.status == 404:
                    raise HTTPException(status_code=401, detail="Invalid token")
                elif response.status == 403:
                    raise HTTPException(status_code=401, detail="Token expired")
                else:
                    logger.error(f"Auth service returned unexpected status {response.status}")
                    raise HTTPException(status_code=500, detail="Authentication service error")

    except asyncio.TimeoutError:
        logger.error("Auth request timed out")
        raise HTTPException(status_code=503, detail="Authorization timeout")

    except aiohttp.ClientError as e:
        logger.warning(f"Auth service connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    except Exception as e:
        logger.exception("Unexpected error in check_token")
        raise HTTPException(status_code=500, detail="Internal auth error")


@router.get("/health", response_model=HealthResponse)
async def health():
    return {'status': 'OK', 'version': config.VERSION}  


async def process_uploading_task(task_id: str, replace=False):
    """Фоновая задача для загрузки данных из файла."""

    logger.info(f"[{task_id}] process_uploading_task started")

    try:
        task = await task_storage.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        # Выполняем загрузку
        result = await data_loader.upload_data_from_file(
            task,
        )

        logger.info(f"[{task_id}] uploading task completed")

        # Обновляем статус
        await task_storage.update_task(task_id, status="READY", progress=100)

        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in data loading task: {e}")

        await task_storage.update_task(task_id, status="ERROR", error=str(e))


async def process_fitting_task(task_id: str):

    logger.info(f"[{task_id}] process_fitting_task started")

    try:
        task = await task_storage.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        model = model_manager.get_model(task.model_id, model_type=task.model_type)
        await model.initialize(parameters=task.fitting_parameters.model_dump())

        result = await model.fit(task.fitting_parameters.data_filter)

        logger.info(f"[{task_id}] fitting task completed")

        await task_storage.update_task(task_id, status="READY", progress=100)

        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in fitting task: {e}")

        await task_storage.update_task(task_id, status="ERROR", error=str(e))


async def process_calculating_fi_task(task_id: str):

    logger.info(f"[{task_id}] process_calculating_fi_task started")

    try:
        task = await task_storage.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        model = model_manager.get_model(task.model_id, model_type=task.model_type)

        await post_processor.calculate_feature_importances(model, task.fitting_parameters.data_filter)

        logger.info(f"[{task_id}] calculating fi task completed")

        await task_storage.update_task(task_id, status="READY", progress=100)

        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in calculating fi task: {e}")

        await task_storage.update_task(task_id, status="ERROR", error=str(e))        


async def get_token_from_header(token: Optional[str] = Header(None, alias="token")):
    """
    Извлекает токен из заголовка запроса.

    Args:
        token: Токен из заголовка X-API-Token

    Returns:
        str: Токен авторизации
    """
    return token


@router.post("/upload_data", response_model=TaskResponse)
async def upload_data(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        replace: bool = Query(default=False)) -> TaskResponse:

    logger.info(f"Starting uploading from file: {file.filename}")

    task_id = str(uuid.uuid4())
    task = await task_storage.create_task(task_id)

    try:
        # Сохраняем файл
        content = await file.read()
        file_path = await task_storage.save_upload_file(task_id, file.filename, content) # type: ignore

        # Обновляем задачу
        await task_storage.update_task(
            task_id,
            type='UPLOAD',
            status="UPLOADING_FILE",
            upload_progress=100,
            file_path=str(file_path),
            replace=replace
        )

        # Запускаем фоновую задачу
        background_tasks.add_task(process_uploading_task, task_id)

        return TaskResponse(task_id=uuid.UUID(task_id), message="Task processing started")

    except Exception as e:
        logger.error(f"Error in uploading task {task_id}: {e}")

        if "File name too long" in str(e):
            await task_storage.update_task(task_id, status="ERROR", error="Имя файла слишком длинное")
            raise HTTPException(
                status_code=500,
                detail="Имя файла слишком длинное. Сократите имя файла и попробуйте загрузить ещё раз"
            )
        else:
            await task_storage.update_task(task_id, status="ERROR", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_status", response_model=Optional[StatusResponse])
async def get_status(
        task_id: str,
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header)
        ) -> StatusResponse:

    status = await task_storage.get_task_status(task_id)

    return status # type: ignore


@router.get("/get_processing_tasks", response_model=list[ProcessingTaskResponse])
async def get_processing_tasks(    
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header)
        ) -> list[ProcessingTaskResponse]:

    result = await task_storage.get_processing_tasks()

    return result 


@router.post("/delete_data", response_model=Optional[StatusResponse])
async def delete_data(     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        data_filter: Optional[dict] = Body(default=None)
        ) -> StatusResponse:
    try:
        await data_loader.delete_data(data_filter)
        return StatusResponse(status='READY', description='All data is deleted successfully')
    except Exception as e:
        logger.error(f"Error in deleting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))  


@router.post("/get_data_count")
async def get_data_count(       
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        data_filter: Optional[dict] = Body(default=None)) -> int:
    try:
        result = await data_loader.get_data_count(data_filter)
        return result
    except Exception as e:
        logger.error(f"Error in getting data count: {e}")
        raise HTTPException(status_code=500, detail=str(e))  


@router.post("/fit")
async def fit(
        background_tasks: BackgroundTasks,   
        model_id: str = Query(), 
        model_type: str = Query(default=''),    
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        parameters: Optional[FittingParameters] = Body(default=None)
        ) -> TaskResponse:
        
    """Запускает обучение модели."""
    logger.info(f"Starting fitting model id: {model_id}")

    task_id = str(uuid.uuid4())
    task = await task_storage.create_task(task_id)

    try:

        # Обновляем задачу
        await task_storage.update_task(
            task_id,
            type='FIT',
            status="PREPARING_DATA",
            model_id=model_id,
            model_type=ModelTypes(model_type) if model_type else None,
            fitting_parameters = parameters
        )

        # Запускаем фоновую задачу
        background_tasks.add_task(process_fitting_task, task_id)

        return TaskResponse(task_id=uuid.UUID(task_id), message="Task processing started")

    except Exception as e:
        logger.error(f"Error in fitting task {task_id}: {e}")

        await task_storage.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))        


@router.post("/calculate_fi")
async def calculate_fi(
        background_tasks: BackgroundTasks,
        model_id: str = Query(),    
        authenticated: bool = Depends(check_token),
        parameters: Optional[FittingParameters] = Body(default=None),        
        token: str = Depends(get_token_from_header)
        ) -> TaskResponse:
        
    logger.info(f"Starting calculating fi model id: {model_id}")

    task_id = str(uuid.uuid4())
    task = await task_storage.create_task(task_id)

    try:
        await task_storage.update_task(
            task_id,
            type='CALCULATE_FI',
            status="PREPARING_DATA",
            model_id=model_id,
            fitting_parameters = parameters
        )
        background_tasks.add_task(process_calculating_fi_task, task_id)

        return TaskResponse(task_id=uuid.UUID(task_id), message="Task processing started")

    except Exception as e:
        logger.error(f"Error in calculating fi task {task_id}: {e}")

        await task_storage.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))       


@router.get("/drop_fi")
async def drop_fi(
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        ) -> str:
    try:
        model = model_manager.get_model(model_id)
        if not model:
            raise ValueError('Model with id "{}" is not defined'.format(model_id))
        
        await post_processor.drop_fi(model)
        
        return 'Model id={} fi dropped sucessfully'.format(model_id)
    except Exception as e:
        logger.error(f"Error in dropping fi: {e}")
        raise HTTPException(status_code=500, detail=str(e))  


@router.post("/predict", response_model=list[RawDataStr])
async def predict(   
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        X: list[RawDataStr] = Body()) -> list[RawDataStr]:

    # try:
        model = model_manager.get_model(model_id)
        
        if not model:
            raise ValueError('Model with id "{}" is not defined'.format(model_id))

        data = []
        for row in X:
            data.append(row.model_dump())

        result_data = await model.predict(data)
        result = []
        for row in result_data:
            result.append(RawDataStr.model_validate(row))
        return  result
    
    # except Exception as e:
    #     logger.error(f"Error in predicting: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))       


@router.get("/get_model_info")
async def get_model_info(
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        ) -> Optional[ModelInfo]:
    try:
        model = model_manager.get_model(model_id)

        if not model:
           return None
        
        result = await model.get_info()

        if result is None:
            return result
        
        return ModelInfo.model_validate(result)
    except Exception as e:
        logger.error(f"Error in getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))  


@router.get("/get_feature_importances")
async def get_feature_importances(
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        ) -> FeatureImportances:
    try:
        model = model_manager.get_model(model_id)
        if not model:
            raise ValueError('Model with id "{}" is not defined'.format(model_id))        
        
        result = await post_processor.get_feature_importances(model)
        
        return FeatureImportances.model_validate(result)
    except Exception as e:
        logger.error(f"Error in getting fi: {e}")
        raise HTTPException(status_code=500, detail=str(e))  


@router.get("/delete_model")
async def delete_model(
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        ) -> str:
    try:
        await model_manager.delete_model(model_id)
        
        return 'Model id={} deleted sucessfully'.format(model_id)
    except Exception as e:
        logger.error(f"Error in deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))      
    

@router.post("/get_sensitivity_analysis")
async def get_sensivity_analysis(
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        sa_data: SAInputData = Body()) -> SAOutputData:
    
    model = model_manager.get_model(model_id)
    if not model:
        raise ValueError('Model with id "{}" is not defined'.format(model_id))

    data_dict = sa_data.model_dump()

    result_data, result_html = await post_processor.get_sa(model, data_dict['data'], 
                                    data_dict['input_indicators'], 
                                    data_dict['output_indicator'],
                                    data_dict['deviations'],
                                    data_dict['auto_selection_number'],
                                    data_dict['get_graph'])
    data = []
    for row in result_data:
        data.append(SARow.model_validate(row))
    return  SAOutputData.model_validate({'data': data, 'graph': result_html})


@router.post("/get_factor_analysis")
async def get_factor_analysis(
        model_id: str = Query(),     
        authenticated: bool = Depends(check_token),
        token: str = Depends(get_token_from_header),
        fa_data: FAInputData = Body()) -> FAOutputData:
    
    model = model_manager.get_model(model_id)
    if not model:
        raise ValueError('Model with id "{}" is not defined'.format(model_id))

    data_dict = fa_data.model_dump()

    result_data, result_html = await post_processor.get_fa(model, 
                                    data_dict['data'], 
                                    data_dict['scenarios'],                                    
                                    data_dict['input_indicators'], 
                                    data_dict['output_indicator'],
                                    data_dict['get_graph'])
    data = []
    for row in result_data:
        data.append(FARow.model_validate(row))
    return  FAOutputData.model_validate({'data': data, 'graph': result_html})