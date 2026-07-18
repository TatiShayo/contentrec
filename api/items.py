from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from starlette.concurrency import run_in_threadpool
from data.items import add_item, get_item, get_all_items
import config

router = APIRouter()

class ItemCreate(BaseModel):
    model_config = {"extra": "forbid"}

    item_id: str = Field(..., min_length=1, max_length=256, description="Unique item identifier", json_schema_extra={"example": "item_42"})
    title: str = Field(..., min_length=1, max_length=1024, description="Title of the item", json_schema_extra={"example": "Introduction to Machine Learning"})
    tags: Optional[str] = Field("", max_length=2048, description="Comma-separated keyword tags", json_schema_extra={"example": "ai,python,tutorial"})
    category: Optional[str] = Field("", max_length=128, description="Primary category of the item", json_schema_extra={"example": "articles"})
    metadata: Optional[dict] = Field(None, description="Optional custom metadata key-value dictionary", json_schema_extra={"example": {"author": "John Doe", "published_year": 2026}})

    @field_validator("metadata")
    @classmethod
    def _limit_metadata_size(cls, v):
        if v is not None and len(v) > 100:
            raise ValueError("metadata may not contain more than 100 keys")
        return v

@router.post("/items")
async def post_item(item: ItemCreate, request: Request):
    await run_in_threadpool(
        add_item, item.item_id, item.title,
        item.tags, item.category, item.metadata
    )
    
    # Incrementally update FAISS index if active
    faiss_index = getattr(request.app.state, 'faiss_index', None)
    if faiss_index is not None:
        item_dict = {
            "item_id": item.item_id,
            "title": item.title,
            "tags": item.tags,
            "category": item.category,
            "metadata": item.metadata
        }
        try:
            await run_in_threadpool(faiss_index.add_item, item_dict)
            await run_in_threadpool(faiss_index.save)
        except Exception as e:
            print(f"Error incrementally adding item to FAISS: {e}")
            
    return {"status": "ok"}

@router.get("/items/{item_id}")
async def read_item(item_id: str):
    item = await run_in_threadpool(get_item, item_id)
    if not item:
        return {"error": "Item not found"}
    return item

@router.get("/items")
async def read_items(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=config.MAX_PAGE_LIMIT),
):
    items = await run_in_threadpool(get_all_items)
    return items[offset : offset + limit]
