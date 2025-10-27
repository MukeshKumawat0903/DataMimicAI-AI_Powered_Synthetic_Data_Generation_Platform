from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from src.core.logger import logger
import os

# Load environment variables
# Load .env for local development only
if os.getenv("RENDER") is None:
    dotenv_path = os.path.abspath('../.env')
    load_dotenv(dotenv_path)

# Load DB URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    # Provide a clear, early error instead of raising AttributeError on .replace
    DATABASE_URL = "postgresql://username:password@host:5432/dbname"
    logger.warning(
        "DATABASE_URL not set in environment; using default placeholder '%s'. "
        "Set the DATABASE_URL env var (or add it to your .env) to point to your database.",
        DATABASE_URL,
    )

# Convert the standard psycopg URL to the SQLAlchemy asyncpg URL only after verifying it's present
DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"ssl": "require"} 
    # connect_args={"ssl": "disable"} # --> For Local
)

AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)
Base = declarative_base()

async def get_async_db():
    async with AsyncSession(engine) as session:
        yield session

# Create the table(s) on startup
async def create_tables():
    """Initializes the database by creating all tables defined in models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)       
