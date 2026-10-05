import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

_fernet = Fernet(os.getenv("TOKEN_ENCRYPTION_KEY").encode())


def encrypt(value: str) -> str:
    return _fernet.encrypt(value.encode()).decode()


def decrypt(value: str) -> str:
    return _fernet.decrypt(value.encode()).decode()
