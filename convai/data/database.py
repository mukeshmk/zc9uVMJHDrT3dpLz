import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from convai.utils.config import settings

logger = logging.getLogger(__name__)

# Create database engine
logger.info(f"Initializing database connection: {settings.DATABASE_URL}")
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
)

logger.debug("Database engine created successfully")

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

logger.debug("Database session factory created successfully")

# Create base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session for FastAPI routes.
    
    Yields:
        Session: Database session object
    """
    logger.debug("Creating new database session")
    db = SessionLocal()
    try:
        yield db
        logger.debug("Database session committed successfully")
    except Exception as e:
        logger.error(f"Database session error: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()
        logger.debug("Database session closed")

def init_db():
    """
    Initialize database tables.
    """
    logger.info("Initializing database tables")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
        raise
