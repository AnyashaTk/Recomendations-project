from copy import deepcopy

import dill
import numpy as np
from data_preprocess import Preprocessing, load_dataset


class LightFM:
    def __init__(self, dataset, path="../../models/lfm_model.dill"):
        with open(path, "rb") as f:
            self.model = dill.load(f)
        self.dataset = deepcopy(dataset)

    def get_vectors(self):
        user_embeddings, item_embeddings = None, None
        if self.model and self.dataset:
            user_embeddings, item_embeddings = self.model.get_vectors(self.dataset)
        _, augmented_item_embeddings = self.augment_inner_product(item_embeddings)
        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        augmented_user_embeddings = np.append(user_embeddings, extra_zero, axis=1)
        return augmented_item_embeddings, augmented_user_embeddings

    @staticmethod
    def augment_inner_product(factors):
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm**2 - normed_factors**2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)
        return max_norm, augmented_factors

    def predict(self, user_id, k=10):
        if self.dataset is None:
            raise ValueError("Dataset not found")
        if self.model is None:
            raise ValueError("Model not found")
        try:
            item_ids = self.model.recommend(
                users=[user_id],
                dataset=self.dataset,
                k=k,
                filter_viewed=False,
            )["item_id"].to_numpy()
            return item_ids
        except KeyError:
            return []


interactions_df, users_df, items_df = load_dataset()
preprocessing_dataset = Preprocessing(users=users_df.copy(), items=items_df.copy(), interactions=interactions_df.copy())
dataset = preprocessing_dataset.get_dataset()

lightfm = LightFM(dataset=dataset)
