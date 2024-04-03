import random
from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError
from service.log import app_logger

from ..models import Bm25KnnModel, UsePopularM, Lightfm, DSSM, CatBoostReranker


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name == "random":
        reco = [random.randint(0, 1000) for _ in range(10)]
    elif model_name == "top":
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
    elif model_name == "knn_bm25":
        knn_model = Bm25KnnModel()
        reco = knn_model(user_id)
    elif model_name == "popular":
        knn_model = UsePopularM()
        reco = knn_model(user_id)
    elif model_name == "lightfm":
        model = Lightfm()
        reco = model(user_id)
    elif model_name == "dssm":
        model = DSSM()
        reco = model(user_id)
    elif model_name == "catboost_ranker":
        model = CatBoostReranker()
        reco = model(user_id)
    else:
        raise ValueError()

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)