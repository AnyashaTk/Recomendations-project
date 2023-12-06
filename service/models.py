import typing as tp

import pandas as pd
from pydantic import BaseModel


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class Bm25KnnModel:
    def __call__(self, user_id: int, *args, **kwargs):
        knn = pd.read_csv("../processed_data/knn_bm25.csv")
        popular = pd.read_csv("../processed_data/popular_10_recs.csv")
        rec = knn[knn["user_id"] == user_id].iloc[:10]["item_id"]
        if len(rec) < 10:
            rec = pd.concat([rec, popular["item_id"].iloc[: 10 - len(rec)]])
        return rec
