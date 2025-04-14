from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = os.path.abspath('../../../.env')
load_dotenv(dotenv_path)

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"ssl": "require"}
)

AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)
Base = declarative_base()

def get_async_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        db.close()