from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header, Depends, Query

import logging
import asyncio
from datetime import datetime, timedelta, timezone
import shutil
import time
import os
from contextlib import asynccontextmanager
from models import model_manager

from api import router

from config import settings, VERSION, DB_URL
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

    try:
        logger.info("Starting DB connection")
        app.db = db_processor
        logger.info(DB_URL)        
        await app.db.connect(url=DB_URL, timeout=5000)
        logger.info("DB connection done")

    except Exception as e:
        logger.error(f"DB startup error: {e}")
        raise 

    try:
        logger.info("Starting models initialize")
        await model_manager.read_models()
        logger.info("Models initializing done")   
    except Exception as e:
        logger.error(f"Models initializing error: {e}")
        raise         

    yield 

    try:
        app.db.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise
   
    logger.info("Shutting down prediction service...")      
        

app = FastAPI(
    title="VBM main API",
    description="API для сервиса авторизации vbm-main",
    lifespan=lifespan,
    version=VERSION
)

# Подключение роутера
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=9085, access_log=False) 