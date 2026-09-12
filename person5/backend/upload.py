from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..models import Image, User
from ..schemas.images import ImageResponse
from .auth import get_current_user
from .config import PROJECT_ROOT
from .database import get_db

router = APIRouter(tags=["images"])
UPLOAD_DIRECTORY = PROJECT_ROOT / "uploads"
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@router.post("/upload", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
	file: UploadFile = File(...),
	current_user: User = Depends(get_current_user),
	db: Session = Depends(get_db),
) -> Image:
	original_filename = Path(file.filename or "").name
	extension = Path(original_filename).suffix.lower()
	if not original_filename or extension not in ALLOWED_EXTENSIONS:
		raise HTTPException(
			status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
			detail="Unsupported image type. Use .tif, .tiff, .png, .jpg, or .jpeg.",
		)

	stored_filename = f"{uuid4().hex}{extension}"
	stored_path = UPLOAD_DIRECTORY / stored_filename
	UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

	try:
		with stored_path.open("wb") as destination:
			while chunk := await file.read(1024 * 1024):
				destination.write(chunk)

		image = Image(
			user_id=current_user.id,
			filename=original_filename,
			file_path=str(stored_path.relative_to(PROJECT_ROOT)),
		)
		db.add(image)
		db.commit()
		db.refresh(image)
		return ImageResponse(id=image.id, filename=image.filename, created_at=image.created_at)
	except Exception:
		if stored_path.exists():
			stored_path.unlink()
		db.rollback()
		raise
	finally:
		await file.close()