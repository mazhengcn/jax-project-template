"""Code to generate processed features."""

import tensorflow as tf

from ..data.pipeline import FeatureDict
from .tf import project_dataset, project_features


def np_data_to_features(raw_data, num_devices=None) -> FeatureDict:
    """Preprocesses NumPy feature dict using TF pipeline."""

    num_examples = raw_data["functions"]["boundary"].shape[0]

    def to_features(x):
        raw_example = {**x, **raw_data["grid"]}
        tensor_dict = project_dataset.np_to_tensor_dict(
            raw_example, raw_data["shape"], project_features.FEATURES.keys()
        )
        if num_devices:

            def reshape_multi_devices(v):
                shape = v.shape
                new_shape = [num_devices, shape[0] // num_devices] + shape[1:]
                return tf.reshape(v, new_shape)

            tensor_dict = tf.nest.map_structure(reshape_multi_devices, tensor_dict)

        return tensor_dict

    dataset = (
        tf.data.Dataset.from_tensor_slices(raw_data["functions"])
        .map(to_features, num_parallel_calls=num_examples)
        .batch(num_examples)
    )
    processed_features = dataset.get_single_element()

    return tf.nest.map_structure(lambda x: x.numpy(), processed_features)
