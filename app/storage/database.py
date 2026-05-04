from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from functools import lru_cache
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(512))
    full_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    revenuecat_customer_id: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
    )
    entitlement_status: Mapped[str] = mapped_column(String(32), default="inactive")
    entitlement_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    current_tier: Mapped[str] = mapped_column(String(32), default="basic")
    clips_used_this_month: Mapped[int] = mapped_column(Integer, default=0)
    quota_reset_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


class AnalysisSessionRecord(Base):
    __tablename__ = "analysis_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("users.id"), index=True)
    overall_status: Mapped[str] = mapped_column(String(32), default="pending")
    bowler_coaching_feedback: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    deliveries: Mapped[list["SessionDeliveryRecord"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class AnalysisJobRecord(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("users.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("analysis_sessions.id"),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255))
    requested_features: Mapped[list[str]] = mapped_column(JSON, default=list)
    overall_status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    feature_results: Mapped[list["AnalysisFeatureResultRecord"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    session_delivery: Mapped["SessionDeliveryRecord | None"] = relationship(
        back_populates="job",
        uselist=False,
    )


class AnalysisFeatureResultRecord(Base):
    __tablename__ = "analysis_feature_results"
    __table_args__ = (UniqueConstraint("job_id", "feature_name", name="uq_job_feature"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_jobs.id"),
        index=True,
    )
    feature_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    job: Mapped["AnalysisJobRecord"] = relationship(back_populates="feature_results")


class SessionDeliveryRecord(Base):
    __tablename__ = "session_deliveries"
    __table_args__ = (UniqueConstraint("session_id", "sequence_no", name="uq_session_sequence"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_sessions.id"),
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_jobs.id"),
        unique=True,
        index=True,
    )
    sequence_no: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    session: Mapped["AnalysisSessionRecord"] = relationship(back_populates="deliveries")
    job: Mapped["AnalysisJobRecord"] = relationship(back_populates="session_delivery")


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(settings.database_url, future=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def init_database() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_database() -> None:
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
