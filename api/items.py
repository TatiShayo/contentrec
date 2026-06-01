from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, List
from starlette.concurrency import run_in_threadpool
from data.items import add_item, get_item, get_all_items

router = APIRouter()

class ItemCreate(BaseModel):
    item_id: str
    title: str
    tags: Optional[str] = ""
    category: Optional[str] = ""
    metadata: Optional[dict] = None

@router.post("/items")
async def post_item(item: ItemCreate):
    await run_in_threadpool(
        add_item, item.item_id, item.title,
        item.tags, item.category, item.metadata
    )
    return {"status": "ok"}

@router.get("/items/{item_id}")
async def read_item(item_id: str):
    item = await run_in_threadpool(get_item, item_id)
    if not item:
        return {"error": "Item not found"}
    return item

@router.get("/items")
async def read_items(offset: int = 0, limit: int = 100):
    items = await run_in_threadpool(get_all_items)
    return items[offset : offset + limit]
