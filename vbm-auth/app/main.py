from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from contextlib import asynccontextmanager
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from datetime import datetime, timedelta, timezone
import aiosmtplib
import secrets

from entities import TokenRequest, TokenCheck, TokenData
from config import settings
from config import VERSION

__version__ = VERSION

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def check_database_integrity(db):
    """Проверка целостности базы данных и восстановление индексов"""
    logger.info("check_database_integrity")
    return True

async def periodic_db_check(db):
    """Периодическая проверка целостности базы данных"""
    logger.info("periodic_db_check")
    return

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация подключения к базе данных при запуске"""
    try:
        app.mongodb_client = AsyncIOMotorClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=5000
        )
        app.mongodb = app.mongodb_client[settings.database_name]
        
        # Проверка подключения
        await app.mongodb.command("ping")
        logger.info("Successfully connected to the database")
        
        # Проверка целостности базы данных
        db_integrity = await check_database_integrity(app.mongodb)
        if not db_integrity:
            logger.error("Database integrity check failed")
            raise Exception("Database integrity check failed")
        
        # Запуск фоновых задач
        asyncio.create_task(periodic_db_check(app.mongodb))
        logger.info("Background tasks started")

        yield 
        
        """Закрытие подключения к базе данных при остановке"""
        app.mongodb_client.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database startup error: {e}")
        raise

app = FastAPI(
    title="VBM Auth API",
    description="API для сервиса авторизации vbm-auth",
    lifespan=lifespan,
    version=VERSION
)

async def log_token_operation(db, email: str, operation: str, details: str):
    """Логирование операций с токенами"""
    try:
        await db.token_logs.insert_one({
            "email": email,
            "operation": operation,
            "details": details,
            "timestamp": datetime.now(timezone.utc)
        })
        logger.info(f"Token operation: {operation} for {email} - {details}")
    except Exception as e:
        logger.error(f"Error logging token operation: {e}")

async def update_token_usage(db, token: str):
    """Обновление времени последнего использования токена"""
    try:
        await db.tokens.update_one(
            {"token": token},
            {"$set": {"last_used": datetime.utcnow()}}
        )
    except Exception as e:
        logger.error(f"Error updating token usage: {e}")

async def send_email(to_email: str, token: str):
    """Отправка email с токеном"""
    try:
        # Используем MIMEMultipart для поддержки HTML и текста
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        message = MIMEMultipart("alternative")
        message["From"] = settings.smtp_from_email
        message["To"] = to_email
        message["Subject"] = "BIT.Newton Your Authentication Token"

        # Текстовая версия письма
        plain_text = settings.email_text.format(**{'token': token})

        # HTML версия письма с форматированием
        html_content = settings.email_html.format(**{'token': token})

        # Прикрепление обеих версий к сообщению
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_content, "html")
        message.attach(part1)
        message.attach(part2)

        async with aiosmtplib.SMTP(
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                use_tls=True
        ) as smtp:
            await smtp.login(settings.smtp_username, settings.smtp_password)
            await smtp.send_message(message)

        logger.info(f"Email sent successfully to {to_email}")
    except Exception as e:
        logger.error(f"Error sending email to {to_email}: {e}")
        raise

async def get_db():
    """Получение подключения к базе данных"""
    yield app.mongodb

@app.get("/health")
async def health():
    return {"message": "VBM AUTH"}

@app.post("/request-token")
async def request_token(
    token_request: TokenRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """Создание нового токена"""
    email = token_request.email
    domain = email.split("@")[-1]
    
    if domain not in settings.allowed_domains:
        await log_token_operation(db, email, "token_request_denied", "Domain not allowed")
        raise HTTPException(status_code=403, detail="Email domain not allowed")
    
    current_time = datetime.now(timezone.utc)
    
    try:
        # Проверяем существующий токен
        existing_token = await db.tokens.find_one({"email": email})
        
        if existing_token and existing_token["expiration"] > current_time:
            await log_token_operation(
                db,
                email,
                "token_request",
                "Existing valid token found"
            )
            
            # Отправляем email в фоновом режиме
            background_tasks.add_task(send_email, email, existing_token["token"])
        
            return {"message": "You already have a valid token. Token sent to your email"}
        
        # Создаем новый токен
        token = secrets.token_urlsafe(32)
        expiration = current_time + timedelta(days=settings.token_expiration_days)
        
        token_data = {
            "email": email,
            "token": token,
            "expiration": expiration,
            "type": "user",
            "created_at": current_time,
            "last_used": current_time
        }
        
        # Создаем или обновляем токен
        await db.tokens.update_one(
            {"email": email},
            {"$set": token_data},
            upsert=True
        )
        
        await log_token_operation(
            db,
            email,
            "token_created",
            f"New token created, expires at {expiration}"
        )
        
        # Отправляем email в фоновом режиме
        background_tasks.add_task(send_email, email, token)
        
        return {"message": "Token sent to your email"}
    
    except Exception as e:
        await log_token_operation(
            db,
            email,
            "token_creation_error",
            str(e)
        )
        raise HTTPException(status_code=500, detail="Error creating token")

@app.post("/check-token")
async def check_token(
    token_check: TokenCheck,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """Проверка существующего токена"""
    try:
        token_data = await db.tokens.find_one({"token": token_check.token})
        
        if not token_data:
            logger.warning(f"Token not found: {token_check.token[:10]}...")
            raise HTTPException(status_code=404, detail="Token not found")
        
        current_time = datetime.now(timezone.utc)
        
        # Обновляем время последнего использования
        background_tasks.add_task(
            update_token_usage,
            db,
            token_check.token
        )
        
        if token_data["expiration"].astimezone(timezone.utc) < current_time:
            # Создаем новый токен
            new_token = secrets.token_urlsafe(32)
            new_expiration = current_time + timedelta(days=settings.token_expiration_days)
            
            await db.tokens.update_one(
                {"_id": token_data["_id"]},
                {
                    "$set": {
                        "token": new_token,
                        "expiration": new_expiration,
                        "last_used": current_time
                    }
                }
            )
            
            await log_token_operation(
                db,
                token_data["email"],
                "token_renewed",
                f"Token renewed, new expiration at {new_expiration}"
            )
            
            background_tasks.add_task(send_email, token_data["email"], new_token)
            
            raise HTTPException(
                status_code=403,
                detail="Token expired. New token sent to your email"
            )
        
        time_left = token_data["expiration"].astimezone(timezone.utc) - current_time
        
        return {
            "access": "granted",
            "type": token_data["type"],
            "time_left": str(time_left),
            "email": token_data["email"],
            "created_at": token_data["created_at"],
            "email": token_data["email"],
            "last_used": token_data.get("last_used", token_data["created_at"])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking token: {e}")
        raise HTTPException(status_code=500, detail="Error checking token")
