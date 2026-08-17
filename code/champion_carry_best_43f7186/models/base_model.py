from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """
    Compatibility shim for repo-benchmark-14days/model_zw/TsingRoc_Baseline.py.

    The native benchmark code imports `models.base_model.BaseModel`. Within the
    Shandong target-day runner, `models` resolves to this local package, so we
    provide the same interface here without modifying the native source file.
    """

    def __init__(self):
        self._stage = None
        self._frozen = False

    def setup(self, stage: str):
        assert stage in ("train", "inference"), f"Invalid stage '{stage}', expected 'train' or 'inference'."
        self._stage = stage
        self._frozen = stage == "inference"

    def _check_trainable(self):
        if self._frozen:
            raise RuntimeError(
                f"Model is frozen (stage={self._stage}). Training or parameter update is not allowed."
            )

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, is_train: bool = True):
        pass

    @abstractmethod
    def train(self, train_data: pd.DataFrame, *args, **kwargs):
        self._check_trainable()

    def fit(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    @abstractmethod
    def predict(self, train_data: pd.DataFrame, *args, **kwargs):
        pass

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
