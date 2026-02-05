from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from backend.app.database import users_collection
from backend.app.core.security import verify_password, get_password_hash, create_access_token
from backend.app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta
import uuid

router = APIRouter(prefix="/auth", tags=["auth"])

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

@router.post("/register", response_model=Token)
async def register(user: UserCreate):
    try:
        if users_collection.find_one({"email": user.email}):
            raise HTTPException(status_code=400, detail="Email already registered")
        if users_collection.find_one({"username": user.username}):
            raise HTTPException(status_code=400, detail="Username already taken")
        hashed_password = get_password_hash(user.password)
        user_id = str(uuid.uuid4())
        user_dict = {
            "user_id": user_id,
            "username": user.username,
            "email": user.email,
            "password": hashed_password,
            "is_verified": False
        }
        users_collection.insert_one(user_dict)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user_id}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "username": user.username}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    db_user = users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user["email"], "user_id": db_user["user_id"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "username": db_user["username"]}

@router.post("/verify")
async def verify_email(email: str):
    # Mock verification for now
    result = users_collection.update_one({"email": email}, {"$set": {"is_verified": True}})
    if result.modified_count == 1:
        return {"message": "Email verified successfully"}
    raise HTTPException(status_code=400, detail="User not found or already verified")
