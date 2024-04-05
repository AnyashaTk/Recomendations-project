import typing as tp

import dill
import pandas as pd
from pydantic import BaseModel
from rectools.dataset import Dataset
from keras.models import load_model

from service.utils.lightfm import LightFM
from service.utils.data_preprocess import Preprocessing, load_dataset


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


def get_data(path="data_original") -> Dataset:
    interactions_df, users_df, items_df = load_dataset(path=path)
    preprocessing_dataset = Preprocessing(
        users=users_df.copy(), items=items_df.copy(), interactions=interactions_df.copy()
    )
    return preprocessing_dataset.get_dataset()


class UsePopularM:
    def __init__(self, path_model="",
                 path_dataset="data_original", **data) -> None:
        with open(path_model + "service/models/model_popular.dill", "rb") as f:
            self.model = dill.load(f)
        self.dataset = get_data(path_dataset)

    def __call__(self, user_id: int, *args, **kwargs):
        return self.model.recommend(users=[0], dataset=self.dataset, k=10, filter_viewed=True)["item_id"].tolist()


class Bm25KnnModel:
    def __init__(self, **data) -> None:
        pass

    def __call__(self, user_id: int, *args, **kwargs) -> list:
        knn = pd.read_csv("../processed_data/knn_bm25.csv")
        popular = pd.read_csv("../processed_data/popular_10_recs.csv")
        rec = knn[knn["user_id"] == user_id].iloc[:10]["item_id"]
        if len(rec) < 10:
            rec = pd.concat([rec, popular["item_id"].iloc[: 10 - len(rec)]])
        return rec


class Lightfm:
    def __init__(self) -> None:
        # with open("service/models/lfm_model.dill", "rb") as f:
        #     self.model = dill.load(f)
        # self.users = pd.read_csv("data_original/users.csv")
        # self.items = pd.read_csv("data_original/items.csv")

        interactions_df, users_df, items_df = load_dataset()
        preprocessing_dataset = Preprocessing(users=users_df.copy(), items=items_df.copy(),
                                              interactions=interactions_df.copy())
        dataset = preprocessing_dataset.get_dataset()

        self.model = LightFM(dataset=dataset)

    def __call__(self, user_id, *args, **kwargs) -> list:
        y_pred = self.model.predict(user_id)  # self.users, self.items)

        return y_pred


class DSSM:

    def __init__(self, path_dataset="data_original"):
        print('model is loading..')
        self.model = load_model("service/models/model_dssm_ep10_l0_13.h5")
        print('dataset is loading..')
        self.dataset = get_data(path_dataset)

    def __call__(self, user_id: int, *args, **kwargs):
        rec = self.model.predict(user_id)
        print(f'user_id: {user_id}, items: {rec}')
        return rec


class CatBoostReranker:
    def __init__(self) -> None:
        with open("service/models/CatBoostRanker_model.dill", "rb") as f:
            self.model = dill.load(f)
        self.data = pd.read_csv("data_original/users.csv")

    def __call__(self, user_id, *args, **kwargs) -> list:
        y_pred = self.model.predict([user_id])

        return y_pred[user_id]
