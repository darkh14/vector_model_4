from pydantic_settings import BaseSettings

with open('email.txt', 'r', encoding='utf-8') as fp:
    EMAIL_TEXT = fp.read()

with open('email.html', 'r', encoding='utf-8') as fp:
    EMAIL_HTML = fp.read()

class Settings(BaseSettings):
    mongodb_url: str
    database_name: str
    allowed_domains: str
    token_expiration_days: int
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_from_email: str
    email_text: str = EMAIL_TEXT
    email_html: str = EMAIL_HTML

    class Config:
        env_file = "../../.env"
        extra = 'allow'
        


settings = Settings()

# Преобразование строки allowed_domains в список
settings.allowed_domains = settings.allowed_domains.split(',')
