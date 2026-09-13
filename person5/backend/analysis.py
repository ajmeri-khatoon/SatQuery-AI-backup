from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Analysis, Execution, Image, Result, User
from ..schemas.analysis import (
    AnalysisCreateResponse,
    AnalysisResultResponse,
    AnalyzeRequest,
    ExecutionResponse,
    QueryRequest,
)
from ..services.analysis_execution import create_contract_analysis, execute_analysis
from .auth import get_current_user
from .config import PROJECT_ROOT
from .database import get_db

router = APIRouter(tags=["analysis"])


def _get_user_image(image_id: int, user_id: int, db: Session) -> Image:
    image = db.scalar(select(Image).where(Image.id == image_id, Image.user_id == user_id))
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return image


def _resolve_images(
    image_id: int | None, image_ids: list[int] | None, user_id: int, db: Session
) -> list[Image]:
    ids = image_ids or ([] if image_id is None else [image_id])
    if not ids or len(set(ids)) != len(ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="one or more image_ids are required and must be unique",
        )
    images = list(
        db.scalars(select(Image).where(Image.user_id == user_id, Image.id.in_(ids))).all()
    )
    by_id = {image.id: image for image in images}
    if len(images) != len(ids):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return [by_id[image_id] for image_id in ids]


def _create_analysis(
    *, images: list[Image], user: User, question: str, requested_capability: str, db: Session
) -> AnalysisCreateResponse:
    analysis = Analysis(user_id=user.id, image_id=images[0].id, question=question, status="pending")
    db.add(analysis)
    db.flush()
    try:
        create_contract_analysis(analysis, images, question, requested_capability, db, PROJECT_ROOT)
    except ValueError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    db.commit()
    db.refresh(analysis)
    return AnalysisCreateResponse(
        analysis_id=analysis.id, status=analysis.status, plan=analysis.plan_data
    )


@router.post("/query", response_model=AnalysisCreateResponse, status_code=status.HTTP_201_CREATED)
def create_query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisCreateResponse:
    images = _resolve_images(request.image_id, request.image_ids, current_user.id, db)
    return _create_analysis(
        images=images,
        user=current_user,
        question=request.question,
        requested_capability=request.requested_capability,
        db=db,
    )


@router.post("/analyze", response_model=AnalysisCreateResponse, status_code=status.HTTP_201_CREATED)
def create_analysis(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisCreateResponse:
    images = _resolve_images(request.image_id, request.image_ids, current_user.id, db)
    return _create_analysis(
        images=images,
        user=current_user,
        question=request.question,
        requested_capability=request.requested_capability,
        db=db,
    )


def _get_user_analysis(analysis_id: int, user_id: int, db: Session) -> Analysis:
    analysis = db.scalar(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user_id)
    )
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


@router.get("/result/{analysis_id}", response_model=AnalysisResultResponse)
def get_result(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResultResponse:
    analysis = _get_user_analysis(analysis_id, current_user.id, db)
    results = db.scalars(
        select(Result).where(Result.analysis_id == analysis.id).order_by(Result.created_at)
    ).all()
    return AnalysisResultResponse(
        analysis_id=analysis.id,
        image_id=analysis.image_id,
        question=analysis.question,
        status=analysis.status,
        results=results,
        plan=analysis.plan_data,
        final_result=analysis.result_data,
        trace=analysis.trace_data,
    )


@router.get("/result/{analysis_id}/mask")
def get_result_mask(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from fastapi.responses import FileResponse
    analysis = _get_user_analysis(analysis_id, current_user.id, db)
    mask_filename = f"{analysis.request_data['request']['id']}_change_mask.png"
    mask_path = PROJECT_ROOT / mask_filename
    if not mask_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask not found or not generated for this analysis")
    return FileResponse(mask_path, media_type="image/png")


@router.post("/analyze/{analysis_id}/run", response_model=AnalysisCreateResponse)
def run_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisCreateResponse:
    analysis = _get_user_analysis(analysis_id, current_user.id, db)
    if analysis.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is already {analysis.status}",
        )
    try:
        analysis = execute_analysis(analysis, db, PROJECT_ROOT)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    return AnalysisCreateResponse(analysis_id=analysis.id, status=analysis.status)


@router.get("/execution/{analysis_id}", response_model=list[ExecutionResponse])
def get_execution(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ExecutionResponse]:
    analysis = _get_user_analysis(analysis_id, current_user.id, db)
    executions = db.scalars(
        select(Execution).where(Execution.analysis_id == analysis.id).order_by(Execution.id)
    ).all()
    return [
        ExecutionResponse(
            id=execution.id,
            step=execution.step or "unknown",
            specialist=execution.specialist,
            status=execution.status,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            error_message=execution.error_message,
            created_at=execution.created_at,
            result_data=execution.result_data,
        )
        for index, execution in enumerate(executions)
    ]
