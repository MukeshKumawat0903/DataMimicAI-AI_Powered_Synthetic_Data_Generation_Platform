from sqlalchemy import Column, String, Integer, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    record_count = Column(Integer)
    sample_data = Column(Text)              # JSON sample preview
    original_data = Column(LargeBinary)     # Raw file (bytes)