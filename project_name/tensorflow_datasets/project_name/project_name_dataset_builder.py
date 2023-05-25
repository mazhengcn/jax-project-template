import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ...data import pipeline
from ...model.tf import project_dataset, project_features

os.environ["NO_GCE_CHECK"] = "true"

tfds.core.utils.gcs_utils._is_gcs_disabled = True


project_features.register_feature(
    "labels", tf.float32, [project_features.NUM_PHASE_COORDS]
)
FEATURES = project_features.FEATURES


def _get_config_names(file):
    config_path = pathlib.Path(__file__).parent / file
    with open(config_path, "r") as f:
        return f.read().splitlines()


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for rte_dataset dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name=name) for name in _get_config_names("CONFIGS.txt")
    ]
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Please download the raw dataset to project_root/data/raw_data
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        feature_info_dict = {}
        for k, v in FEATURES.items():
            _, shape = v
            new_shape = [None if isinstance(k, str) else k for k in shape]
            feature_info_dict[k] = tfds.features.Tensor(
                shape=new_shape, dtype=np.float32, encoding="zlib"
            )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(feature_info_dict),
            metadata=self.metadata_dict,
            homepage="https://github.com/xxx/xxx.git",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        datasets_dir = dl_manager.manual_dir / self.builder_config.name
        filenames = [
            file.name for file in datasets_dir.iterdir() if file.name.endswith(".mat")
        ]

        data_pipeline = pipeline.DataPipeline(datasets_dir, filenames)
        raw_data = data_pipeline.process(normalization=True)

        self.metadata_dict.update(
            {
                "normalization": tf.nest.map_structure(
                    lambda x: str(x), raw_data["normalization"]
                )
            }
        )

        return {
            "train": self._generate_examples(raw_data),
        }

    def _generate_examples(self, raw_data):
        """Yields examples."""

        num_examples = raw_data["shape"]["num_examples"]

        for i in range(num_examples):
            np_example = {
                **tf.nest.map_structure(lambda x: x[i], raw_data["functions"]),
                **raw_data["grid"],
            }
            tensor_dict = project_dataset.np_to_tensor_dict(
                np_example, raw_data["shape"], FEATURES.keys()
            )
            yield i, tf.nest.map_structure(lambda x: x.numpy(), tensor_dict)
