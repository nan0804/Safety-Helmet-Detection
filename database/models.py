"""
Database Models
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base     = declarative_base()
_engine  = None
_Session = None


class Worker(Base):
    __tablename__ = 'workers'
    id          = Column(Integer, primary_key=True)
    worker_id   = Column(String(50), unique=True, nullable=False)
    name        = Column(String(100), nullable=False)
    designation = Column(String(100))
    department  = Column(String(100))
    phone       = Column(String(20))
    face_path   = Column(String(255))
    registered  = Column(DateTime, default=datetime.utcnow)
    is_active   = Column(Boolean, default=True)

    def to_dict(self):
        return {
            'id':          self.id,
            'worker_id':   self.worker_id,
            'name':        self.name,
            'designation': self.designation,
            'department':  self.department,
            'phone':       self.phone,
            'registered':  self.registered.isoformat() if self.registered else None,
            'is_active':   self.is_active,
        }


class Violation(Base):
    __tablename__ = 'violations'
    id              = Column(Integer, primary_key=True)
    worker_id       = Column(String(50))
    worker_name     = Column(String(100), default='Unknown')
    violation_type  = Column(String(50),  default='No Helmet')
    camera_id       = Column(String(50))
    confidence      = Column(Float)
    face_confidence = Column(Float)
    image_path      = Column(String(255))
    timestamp       = Column(DateTime, default=datetime.utcnow)
    is_resolved     = Column(Boolean, default=False)
    notes           = Column(Text)

    def to_dict(self):
        return {
            'id':              self.id,
            'worker_id':       self.worker_id,
            'worker_name':     self.worker_name,
            'violation_type':  self.violation_type,
            'camera_id':       self.camera_id,
            'confidence':      round(self.confidence * 100, 1) if self.confidence else 0,
            'face_confidence': round(self.face_confidence, 1) if self.face_confidence else 0,
            'image_path':      self.image_path,
            'timestamp':       self.timestamp.isoformat() if self.timestamp else None,
            'is_resolved':     self.is_resolved,
            'notes':           self.notes,
        }


def init_db(db_url):
    global _engine, _Session
    db_file = db_url.replace('sqlite:///', '')
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    _engine  = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine)
    return _engine, _Session


def get_session(db_url=None):
    global _engine, _Session
    if _Session is None:
        if db_url is None:
            raise ValueError("db_url required on first call")
        init_db(db_url)
    return _Session()
