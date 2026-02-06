from backend.app.core.config import SECRET_KEY, ALGORITHM, OPENAI_API_KEY
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from backend.app.database import users_collection

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_client = None
_quota_exhausted = False

def is_openai_ready() -> bool:
    return _client is not None

def get_openai_client():
    global _client
    if _client is None:
        raise RuntimeError("OpenAI client not initialized. Please update the API key.")
    return _client

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

def mark_quota_exhausted():
    global _quota_exhausted
    _quota_exhausted = True

def update_openai_key(new_key: str):
    global _client, _quota_exhausted
    import httpx
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=new_key, http_client=httpx.Client(trust_env=False))
        _quota_exhausted = False
        return True
    except Exception as e:
        print(f"Error updating OpenAI key: {e}")
        return False

def init_openai_from_env():
    global _client, _quota_exhausted
    try:
        if OPENAI_API_KEY:
            from openai import OpenAI
            _client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(trust_env=False))
            _quota_exhausted = False
            return True
        return False
    except Exception as e:
        print(f"Error initializing OpenAI from env: {e}")
        return False
