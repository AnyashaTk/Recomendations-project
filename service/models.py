import typing as tp

from pydantic import BaseModel
import pickle
import pandas as pd


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class Bm25KnnModel(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self.knn = pd.read_csv('../processed_data/knn_bm25.csv')
        self.popular = pd.read_csv('../processed_data/popular_10_recs.csv')

    def __call__(self, user_id: int, *args, **kwargs):
        print(user_id)
        rec = self.knn[self.knn['user_id'] == user_id].iloc[:10]['item_id']
        if len(rec) < 10:
            rec = pd.concat([rec, self.popular['item_id'].iloc[:10-len(rec)]])
        return rec

