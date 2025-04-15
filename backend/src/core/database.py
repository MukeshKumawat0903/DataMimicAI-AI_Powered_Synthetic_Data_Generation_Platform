from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

# Load environment variables
# Load .env for local development only
if os.getenv("RENDER") is None:
    dotenv_path = os.path.abspath('../../../.env')
    load_dotenv(dotenv_path)

# Load DB URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

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
