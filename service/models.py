import dill
import typing as tp

import pandas as pd
from pydantic import BaseModel

from service.utils.data_preprocess import Preprocessing, load_dataset


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


def get_data(path="data_original"):
    interactions_df, users_df, items_df = load_dataset(path=path)
    preprocessing_dataset = Preprocessing(
        users=users_df.copy(), items=items_df.copy(), interactions=interactions_df.copy()
    )
    return preprocessing_dataset.get_dataset()


class UsePopularM(BaseModel):
    def __init__(self, path_model="", path_dataset="../data_original", **data):
        super().__init__(**data)
        with open(path_model + "model_popular.dill", "rb") as f:
            self.model = dill.load(f)
        self.dataset = get_data(path_dataset)

    def __call__(self, user_id: int, *args, **kwargs):
        return self.model.recommend(users=[0], dataset=self.dataset, k=10, filter_viewed=True)["item_id"].tolist()


class Bm25KnnModel(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)

    def __call__(self, user_id: int, *args, **kwargs):
        knn = pd.read_csv("../processed_data/knn_bm25.csv")
        popular = pd.read_csv("../processed_data/popular_10_recs.csv")
        rec = knn[knn["user_id"] == user_id].iloc[:10]["item_id"]
        if len(rec) < 10:
            rec = pd.concat([rec, popular["item_id"].iloc[: 10 - len(rec)]])
        return rec


class CatBoostReranker(BaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.model = dill.load(open("../models/CatBoostRanker_model.dill", "rb"))
        self.data = pd.read_csv("../data_original/users.csv")

    def __call__(self, user_id, *args, **kwargs):
        y_pred = self.model.predict(self.data)

        return ""
