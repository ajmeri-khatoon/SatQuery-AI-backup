from datetime import datetime

from pydantic import BaseModel


class ImageResponse(BaseModel):
    id: int
    filename: str
    file_path: str | None = None
    created_at: datetime
