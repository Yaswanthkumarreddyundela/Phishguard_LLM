# modules/database.py
from sqlalchemy import (
    create_engine, Column, String, Float, Boolean,
    DateTime, Integer, Text, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, Dict
from config.settings import config

Base = declarative_base()


class EmailDetectionResult(Base):
    """
    Database schema for storing detection results.
    
    Design decisions:
    - email_hash as primary key: natural deduplication
    - Store probability + risk_score: different uses
      (probability for ML analysis, risk_score for display)
    - label as string: human readable, easy to query
    - explanation stored as text: audit trail
    - processed_at: for time-series analysis of phishing trends
    """
    __tablename__ = "email_detections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email_hash = Column(String(64), unique=True, nullable=False, index=True)
    is_phishing = Column(Boolean, nullable=False)
    label = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    explanation = Column(Text, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "email_hash": self.email_hash,
            "is_phishing": self.is_phishing,
            "label": self.label,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "explanation": self.explanation,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }


class DatabaseManager:
    """
    SQLite database manager using SQLAlchemy ORM.
    
    Why SQLite for this project?
    - Zero configuration
    - File-based (easy backup/migration)
    - Sufficient for < 1M emails
    - Easy to upgrade to PostgreSQL later
      (just change DATABASE_URL in settings)
    """
    
    def __init__(self):
        self.engine = create_engine(
            config.DATABASE_URL,
            connect_args={"check_same_thread": False}  # Required for SQLite + FastAPI
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def init_db(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        print("[DB] Database initialized")
    
    def store_result(
        self,
        email_hash: str,
        is_phishing: bool,
        confidence: float,
        risk_score: float,
        label: str,
        explanation: str
    ) -> bool:
        """
        Store detection result.
        Uses upsert pattern — if email already processed, update it.
        """
        db = self.SessionLocal()
        try:
            # Check for existing record
            existing = db.query(EmailDetectionResult).filter(
                EmailDetectionResult.email_hash == email_hash
            ).first()
            
            if existing:
                # Update existing
                existing.is_phishing = is_phishing
                existing.confidence = confidence
                existing.risk_score = risk_score
                existing.label = label
                existing.explanation = explanation
                existing.processed_at = datetime.utcnow()
            else:
                # Insert new
                record = EmailDetectionResult(
                    email_hash=email_hash,
                    is_phishing=is_phishing,
                    label=label,
                    confidence=confidence,
                    risk_score=risk_score,
                    explanation=explanation
                )
                db.add(record)
            
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"[DB] Error storing result: {e}")
            return False
        finally:
            db.close()
    
    def get_result(self, email_hash: str) -> Optional[Dict]:
        """Retrieve result by email hash."""
        db = self.SessionLocal()
        try:
            record = db.query(EmailDetectionResult).filter(
                EmailDetectionResult.email_hash == email_hash
            ).first()
            return record.to_dict() if record else None
        finally:
            db.close()
    
    def get_stats(self) -> Dict:
        """
        Aggregate statistics.
        Useful for dashboard and monitoring.
        """
        db = self.SessionLocal()
        try:
            total = db.query(func.count(EmailDetectionResult.id)).scalar()
            phishing_count = db.query(func.count(EmailDetectionResult.id)).filter(
                EmailDetectionResult.is_phishing == True
            ).scalar()
            avg_confidence = db.query(
                func.avg(EmailDetectionResult.confidence)
            ).scalar() or 0
            
            recent_phishing = db.query(EmailDetectionResult).filter(
                EmailDetectionResult.is_phishing == True
            ).order_by(
                EmailDetectionResult.processed_at.desc()
            ).limit(10).all()
            
            return {
                "total_processed": total,
                "phishing_detected": phishing_count,
                "legitimate": total - phishing_count,
                "phishing_rate": round(phishing_count / total * 100, 2) if total > 0 else 0,
                "average_confidence": round(avg_confidence, 4),
                "recent_phishing": [r.to_dict() for r in recent_phishing]
            }
        finally:
            db.close()
    
    def get_history(self, limit: int = 100, phishing_only: bool = False) -> list:
        """Get detection history."""
        db = self.SessionLocal()
        try:
            query = db.query(EmailDetectionResult)
            if phishing_only:
                query = query.filter(EmailDetectionResult.is_phishing == True)
            records = query.order_by(
                EmailDetectionResult.processed_at.desc()
            ).limit(limit).all()
            return [r.to_dict() for r in records]
        finally:
            db.close()