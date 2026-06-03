"""Pydantic request/response models for dashboard API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FilterState(BaseModel):
    date_from: Optional[int] = None
    date_to: Optional[int] = None
    search: Optional[str] = None


class PostsRequest(FilterState):
    limit: int = 20
    offset: int = 0
    sentiment: Optional[str] = None
    week: Optional[str] = None
    entity_text: Optional[str] = None
    post_id: Optional[str] = None


class EvidenceRequest(FilterState):
    topic_id: Optional[int] = None


class QARequest(BaseModel):
    question_id: str


class SearchMeta(BaseModel):
    match_type: str = "none"
    matched_entity: Optional[str] = None
    matched_entity_count: Optional[int] = None


class EntityItem(BaseModel):
    text: str
    label: str
    mention_count: int
    post_count: int
    sentiment_breakdown: dict[str, int]
    dominant_sentiment: str


class TopicItem(BaseModel):
    topic_id: int
    label: str
    keywords: list[str]
    post_count: int
    sentiment_breakdown: dict[str, int]
    dominant_sentiment: str


class InfluencerItem(BaseModel):
    rank: int
    entity: str
    entity_label: str
    mention_count: int
    engagement_score: float
    dominant_sentiment: str
    top_topics: list[str] = Field(default_factory=list)


class PostItem(BaseModel):
    post_id: str
    title: str
    clean_text: str
    score: int
    num_comments: int
    author: str
    created_utc: int
    sentiment: str
    sentiment_confidence: float
    entities: list[dict[str, str]]
    url: str
