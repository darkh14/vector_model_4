from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header, Depends, Query

import logging
import asyncio
from datetime import datetime, timedelta, timezone
import shutil
import time
import os
from uuid import uuid4

from app.entities import LoadingDataResponse
from config import settings, VERSION
from app.auth import check_token

__version__ = VERSION

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VBM Auth API",
    description="API для сервиса авторизации vbm-auth",
    # lifespan=lifespan,
    version=VERSION
)



@app.get("/health")
async def health():
    return {"message": "VBM"}

@app.get("/get_version")
async def get_version():
    return {"version": VERSION}


@app.post("/load_data", response_model=LoadingDataResponse)
async def load_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    authenticated: bool = Depends(check_token),
    loading_id: str = Query(default='', description="ID of loading task. Must be unique id")
    ) -> None:

    start_time = time.time()
    logger.info(f"Data loading started")

    if not loading_id:
        loading_id = uuid4()

    file_path = os.path.join(settings.source_folder, f"{loading_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"File saved: {file_path}") 
    
    return LoadingDataResponse(loading_id=loading_id, message='Data loaded sucessfully')   