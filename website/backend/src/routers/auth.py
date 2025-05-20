from fastapi import APIRouter, status, Depends, HTTPException, Response
from src.schemas import UserRead, UserCreate, UserLogin
from sqlalchemy.orm import Session
from src.dependencies import get_db, get_current_user
from src.database import SessionLocal
from src.models import User
from src.utils import get_password_hash, verify_password, create_access_token
from datetime import timedelta

router = APIRouter(
    prefix='/auth',
    tags=['Authentication']
)

@router.post('/register', response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail='Имя пользователя или email уже зарегестрированы')
    
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post('/login')
def login(user: UserLogin, db: Session = Depends(get_db), response: Response = None):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail='Неправильные учетные данные')
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={'sub': db_user.username},
        expires_delta=access_token_expires
    )
    
    response.set_cookie(
        key='auth_token',
        value=access_token,
        httponly=True,
        max_age=1800,
        expires=1800,
        secure=False,  
        samesite='strict'
    )

    return {"status": True}

@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Успешный выход из системы"}

@router.post('/verify-token', dependencies=[Depends(get_current_user)])
def verify_token():
    return {"valid": True}