from sqlalchemy import Table, Column, Integer, String, DateTime
from src.database import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False, index=True)

class Vehicle(Base):
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    plate_number = Column(String, nullable=False, index=True)
    entry_time = Column(DateTime, nullable=False)