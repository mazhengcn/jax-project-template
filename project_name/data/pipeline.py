from collections.abc import Mapping
from typing import Optional

import numpy as np

RawFeatureDict = Mapping[str, Mapping[str, np.ndarray]]
FeatureDict = Mapping[str, np.ndarray]


def make_data_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    pass


def make_grid_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    pass


def make_shape_dict(np_data: Mapping[str, np.ndarray]) -> Mapping[str, int]:
    pass


class DataPipeline:
    def __init__(self, source_dir: str, data_name_list: list[str]):
        self.source_dir = source_dir
        self.data_name_list = data_name_list
        self.data = self.load_data()

    def load_data(self):
        pass

    def process(self, normalization: Optional[bool] = False) -> RawFeatureDict:
        data_feature = make_data_features(self.data)
        grid_feature = make_grid_features(self.data)
        shape_dict = make_shape_dict(self.data)

        raw_data = {
            "functions": data_feature,
            "grid": grid_feature,
            "shape": shape_dict,
        }

        return raw_data
