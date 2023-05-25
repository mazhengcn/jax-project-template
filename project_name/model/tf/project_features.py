import enum
from collections.abc import Sequence
from typing import Optional

import tensorflow as tf

# Type aliases.
FeaturesMetadata = dict[str, tuple[tf.dtypes.DType, Sequence[str | int]]]


class FeatureType(enum.Enum):
    ZERO_DIM = 0
    ONE_DIM = 1
    TWO_DIM = 2


NUM_DIM = 2
# Placeholder values that will be replaced with their true value at runtime.
NUM_POSITION_COORDS = "num position coordinates placeholder"
NUM_VELOCITY_COORDS = "num velocity coordinates placeholder"
NUM_PHASE_COORDS = "num phase coordinates placeholder"
NUM_BOUNDARY_COORDS = "num boundary coordinates placeholder"

FEATURES = {
    # Static features of RTE #
    "phase_coords": (tf.float32, [NUM_PHASE_COORDS, 2 * NUM_DIM]),
    "boundary_coords": (tf.float32, [NUM_BOUNDARY_COORDS, 2 * NUM_DIM]),
    "boundary_weights": (tf.float32, [NUM_BOUNDARY_COORDS]),
    "position_coords": (tf.float32, [NUM_POSITION_COORDS, NUM_DIM]),
    "velocity_coords": (tf.float32, [NUM_VELOCITY_COORDS, NUM_DIM]),
    "velocity_weights": (tf.float32, [NUM_VELOCITY_COORDS]),
    "boundary": (tf.float32, [NUM_BOUNDARY_COORDS]),
    "some_functions_1": (tf.float32, [NUM_POSITION_COORDS, 3]),
    "some_functions_2": (tf.float32, [NUM_PHASE_COORDS, 1]),
    "some_parameter": (tf.int32, [NUM_DIM]),
}

FEATURE_TYPES = {k: v[0] for k, v in FEATURES.items()}
FEATURE_SIZES = {k: v[1] for k, v in FEATURES.items()}

# Extra features for training
# "boundary_scattering_kernel": (
#     tf.float32,
#     [NUM_BOUNDARY_COORDS, NUM_VELOCITY_COORDS],
# ),
# "psi_label": (tf.float32, [NUM_PHASE_COORDS]),


def register_feature(name: str, type_: tf.dtypes.DType, shape_: tuple[str | int]):
    """Register extra features used in custom datasets."""
    FEATURES[name] = (type_, shape_)
    FEATURE_TYPES[name] = type_
    FEATURE_SIZES[name] = shape_


def shape(
    feature_name: str,
    num_position_coords: int | None = None,
    features: Optional[FeaturesMetadata] = None,
):
    """Get the shape for the given feature name.

    Args:
      feature_name: String identifier for the feature.
      features: A feature_name to (tf_dtype, shape) lookup;
        defaults to FEATURES.

    Returns:
      List of ints representation the tensor size.

    Raises:
      ValueError: If a feature is requested but no concrete
        placeholder value is given.
    """
    features = features or FEATURES

    unused_dtype, raw_sizes = features[feature_name]

    replacements = {
        NUM_POSITION_COORDS: num_position_coords,
    }

    sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError(
                "Could not parse %s (shape: %s) with values: %s"
                % (feature_name, raw_sizes, replacements)
            )
    return sizes
