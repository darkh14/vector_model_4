from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel 
from datetime import datetime, timedelta, UTC

import logging
from typing import Optional
import asyncio
from contextlib import asynccontextmanager
from passlib.context import CryptContext

from jose import jwt
from datetime import datetime, timedelta, timezone

from config import (DB_URL, 
                    AUTH_DATABASE_NAME, 
                    SECRET_KEY, 
                    ENCODING_ALHORYTHM, 
                    TOKEN_EXPIRATION_DAYS, 
                    AUTH_ADMIN_PASSWORD, 
                    VERSION)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenRequest(BaseModel):
    user: str


class TokenCheck(BaseModel):
    token: str


class TokenData(BaseModel):
    user: str
    token: str
    expiration: datetime
    created_at: datetime
    last_used: Optional[datetime]


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Starting DB connection")

        app.db_client = AsyncIOMotorClient(
            DB_URL,
            serverSelectionTimeoutMS=5000
        )
        app.db = app.db_client[AUTH_DATABASE_NAME]

        await app.db.command("ping")

        # Проверка целостности базы данных
        db_integrity = await check_database_integrity(app.db)
        if not db_integrity:
            logger.error("Database integrity check failed")
            raise Exception("Database integrity check failed")
        
        # Запуск фоновых задач
        asyncio.create_task(cleanup_expired_tokens(app.db))
        asyncio.create_task(periodic_db_check(app.db))        
        logger.info("DB connection done")

        existing_user = await app.db.users.find_one({"user": 'admin'})
        if not existing_user:
            await user_manager.set_user('admin', AUTH_ADMIN_PASSWORD, AUTH_ADMIN_PASSWORD, app.db, do_not_check=True)
            

    except Exception as e:
        logger.error(f"DB startup error: {e}")
        raise       

    yield 

    try:
        app.db_client.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise
   
    logger.info("Shutting down prediction service...")    


app = FastAPI(
    title="VBM Auth API",
    description="API для сервиса авторизации bshp-auth",
    version=VERSION,
    lifespan=lifespan
)


async def check_database_integrity(db):
    return True


async def periodic_db_check(db):
    return


async def log_token_operation(db, user: str, operation: str, details: str):
    try:
        await db.token_logs.insert_one({
            "user": user,
            "operation": operation,
            "details": details,
            "timestamp": datetime.now(tz=UTC)
        })
        logger.info(f"Token operation: {operation} for {user} - {details}")
    except Exception as e:
        logger.error(f"Error logging token operation: {e}")


async def cleanup_expired_tokens(db):
    return


async def get_db():
    yield app.db


class UserManager:

    def __init__(self):
        pass

    async def set_user(self, user, password, admin_password, db, do_not_check=False):
    
        if not do_not_check:
            check_result = await self._check_user('admin', admin_password, db)
            if not check_result:
                return ['', '', '']
        
        existing_user = await db.users.find_one({"user": user})
        current_time = datetime.now(tz=UTC)

        expiration = current_time + timedelta(days=TOKEN_EXPIRATION_DAYS)
        token = self._create_access_token({'user': user, 'exp': expiration})        
        if existing_user:

            await db.users.update_one({"user": user},                 
                                {
                                "$set": {
                                "user": user,
                                "token": token,
                                "password": self._get_password_hash(password),
                                "expiration": expiration,
                                "last_used": current_time
                                }
                                })
            operation = ('update', 'updating', 'updated')            
        else:
            await db.users.insert_one({"user": user,
                            "token": token,
                            "password": self._get_password_hash(password),
                            "expiration": expiration,
                            "last_used": current_time,
                            "created_at": current_time
                            })
            operation = ('add', 'adding', 'added')
            
        await log_token_operation(
            db,
            user,
            "User_{}".format(operation[0]),
            f'User {operation[0]} "{user}"'
        )

        return operation       

    async def change_password(self, user, password, old_password, db):
        existing_user = await db.users.find_one({"user": user})
        
        if not existing_user:
            raise ValueError('User "{}" not found'.format(user))
        
        if not await self._check_user(user, old_password, db):
            raise ValueError('User "{}" wrong password'.format(user))   

        await db.users.update_one({"user": user},                 
                    {
                    "$set": {
                    "user": user,
                    "token": existing_user['token'],
                    "password": self._get_password_hash(password),
                    "expiration": existing_user['expiration'],
                    "last_used": existing_user['last_used']
                    }
                    })

        return 'password is changed sucessfully'

    async def delete_user(self, user, admin_password, db):
  
        existing_user = await db.users.find_one({"user": user})
        
        if not existing_user:
            raise ValueError('User "{}" not found'.format(user))
        
        if not await self._check_user('admin', admin_password, db):
            raise ValueError('Wrong admin password'.format(user))   

        await db.users.delete_one({"user": user})

        await log_token_operation(
            db,
            user,
            "User_delete",
            f'User delete "{user}"'
        )    

        return 'user deleted sucessfully'
   
    async def request_token(self, user, password, db):
        existing_user = await db.users.find_one({"user": user})
        
        if not existing_user:
            raise ValueError('User "{}" not found'.format(user))
        
        if not await self._check_user(user, password, db):
            raise ValueError('User "{}" wrong password'.format(user))   
        
        current_time = datetime.now(tz=UTC)

        token = existing_user['token']
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ENCODING_ALHORYTHM]) 
        expiration = payload['exp']   
        expiration = datetime.fromtimestamp(int(expiration), tz=UTC)
        if not expiration or current_time > expiration:

            expiration = current_time + timedelta(days=TOKEN_EXPIRATION_DAYS)
            token = self._create_access_token({'user': user, 'exp': expiration})
            db.users.update_one({"user": user},                 
                {
                "$set": {
                "user": user,
                "token": token,
                "password": self._get_password_hash(password),
                "expiration": expiration,
                "last_used": current_time
                }
                })

        await log_token_operation(
            db,
            user,
            "token_requested",
            f'Token requested for user "{user}", expires at {expiration}'
        )
        
        return token        
                        
    async def check_token(self, token, background_tasks, db):
        
        token_data = await db.users.find_one({"token": token})

        if not token_data:
            logger.warning(f"Token not found: {token[:10]}...")
            raise HTTPException(status_code=404, detail="Token not found")
        
        current_time = datetime.now(tz=UTC)
        
        background_tasks.add_task(
            self._update_token_usage,
            db,
            token
        )
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ENCODING_ALHORYTHM]) 
        expiration = payload['exp']
        expiration = datetime.fromtimestamp(int(expiration), tz=UTC)   

        if expiration < current_time:

            expiration = current_time + timedelta(days=TOKEN_EXPIRATION_DAYS)
            token = self._create_access_token({'user': payload['user'], 'exp': expiration})

            db.users.update_one({"_id": token_data['_id']},                 
                {
                "$set": {
                "token": token,
                "expiration": expiration,
                "last_used": current_time
                }
                })
            
            await log_token_operation(
                db,
                token_data["user"],
                "token_renewed",
                f"Token renewed, new expiration at {expiration}"
            )
            
            raise HTTPException(
                status_code=403,
                detail="Token expired. Request new token!"
            )
        
        time_left = expiration - current_time
        
        await log_token_operation(
            db,
            token_data["user"],
            "token_checked",
            f'Token checked for user "{token_data["user"]}"'
        )
        return {
            "access": "granted",
            "time_left": str(time_left),
            "user": token_data["user"],
            "created_at": token_data["created_at"],
            "last_used": token_data.get("last_used", token_data["created_at"])
        }

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    async def _check_user(self, user, password, db):
        existing_user = await db.users.find_one({"user": user})

        if not existing_user:
            raise HTTPException(status_code=500, detail='User "{}" not found'.format(user))         
        
        if not self._verify_password(password, existing_user['password']):
            raise HTTPException(status_code=500, detail='Wrong password')

        return existing_user  

    def _create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        if 'exp' not in data:
            expire = datetime.now(timezone.utc) + timedelta(days=30)
            to_encode.update({"exp": expire})
        encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ENCODING_ALHORYTHM)
        return encode_jwt

    def _get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    async def _update_token_usage(self, db, token: str):

        try:
            await db.users.update_one(
                {"token": token},
                {"$set": {"last_used": datetime.now(tz=UTC)}}
            )
        except Exception as e:
            logger.error(f"Error updating token usage: {e}")


@app.get('/')
async def main_page():
    """
    Root method returns html ok description
    @return: HTML response with ok micro html
    """
    return HTMLResponse('<h2>VBM auth service</h2> <br> <h3>Connection established</h3>')


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "OK", "version": VERSION}


@app.get("/set_user")
async def set_user(user: str,
                   password: str,
                   admin_password: str,
                   db = Depends(get_db)
                   ) -> str:
          
    try:
        await user_manager.set_user(user, password, admin_password, db)
    except Exception as e:
        await log_token_operation(
            db,
            user,
            f"Error adding/updating setting user",
            'Error adding/updating user "{}". {}'.format(user, str(e))
        )        
        raise HTTPException(status_code=500, detail='Error adding/updating user "{}". {}'.format(user, str(e)))        

    return 'user adding/updating sucessfully'


@app.get("/change_password")
async def change_password(user: str,
                   password: str, 
                   old_password: str,
                   db = Depends(get_db)
                   ) -> str:

    try:
        result = await user_manager.change_password(user, password, old_password, db)
   
    except Exception as e:
        await log_token_operation(
            db,
            user,
            f"Error changing password",
            'Error changing password of user "{}". {}'.format(user, str(e))
        )        
        raise HTTPException(status_code=500, detail='Error changing password of user "{}". {}'.format(user, str(e)))        

    return result


@app.get("/delete_user")
async def delete_user(user: str,
                   password,
                   db = Depends(get_db)
                   ) -> str:
          
    try:
        result = await user_manager.delete_user(user, password, db)
    except Exception as e:
        await log_token_operation(
            db,
            user,
            "Error adding deleting",
            'Error adding deleting "{}". {}'.format(user, str(e))
        )        
        raise HTTPException(status_code=500, detail='Error deleting user "{}". {}'.format(user, str(e)))        

    return result


@app.get("/get_token")
async def request_token(user: str,
                        password: str,
                        db = Depends(get_db)
                        ) -> str:
    
    try:
        token = await user_manager.request_token(user, password, db)
        return token
    
    except Exception as e:
        await log_token_operation(
            db,
            user,
            "token_requesting_error",
            str(e)
        )
        raise HTTPException(status_code=500, detail="Error requesting token. {}".format(str(e)))


@app.post("/check_token")
async def check_token(
    token_check: TokenCheck,
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
    ):
    try:
        result = await user_manager.check_token(token_check.token, background_tasks, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking token: {e}")
        raise HTTPException(status_code=500, detail="Error checking token")
    
    return result


user_manager = UserManager()