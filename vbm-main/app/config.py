from pydantic_settings import BaseSettings
import os

VERSION = '1.0.0.0'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    source_folder: str = os.path.join(BASE_DIR, 'data', 'source')
    class Config:
        env_file = "../../.env"
        extra = 'allow'
        
settings = Settings()

