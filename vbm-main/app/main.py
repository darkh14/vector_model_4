from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header, Depends, Query

import logging
import asyncio
from datetime import datetime, timedelta, timezone
import shutil
import time
import os
from contextlib import asynccontextmanager

from api import router

from config import settings, VERSION, MONGODB_URL
from auth import check_token
from db import db_processor

__version__ = VERSION

# Настройка логирования
logging.getLogger("VBM_logger").setLevel(logging.ERROR)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting transcription service...")
    # """Инициализация подключения к базе данных при запуске"""
    try:
        app.db = db_processor
        await app.db.connect(url=MONGODB_URL, timeout=5000)


    except Exception as e:
        logger.error(f"Database startup error: {e}")
        raise    

    yield 
        
    try:
        """Закрытие подключения к базе данных при остановке"""
        app.db.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise
   
    logger.info("Shutting down transcription service...")
        

app = FastAPI(
    title="VBM Auth API",
    description="API для сервиса авторизации vbm-auth",
    lifespan=lifespan,
    version=VERSION
)

# Подключение роутера
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=9085, access_log=False) 