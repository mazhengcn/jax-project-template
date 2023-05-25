import enum
import os
from collections.abc import Generator, Mapping, Sequence
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

FeatureDict = Mapping[str, np.ndarray]


class Split(enum.Enum):
    TRAIN = 1
    TRAIN_AND_VALID = 2
    VALID = 3
    TEST = 4


def load(
    name: str,
    *,
    split: Split,
    split_percentage: str = "",
    tfds_dir: str | os.PathLike | None = None,
    is_training: bool = False,
    # batch_sizes should be:
    # [device_count, per_device_outer_batch_size]
    # total_batch_size = device_count * per_device_outer_batch_size
    batch_sizes: Optional[Sequence[int] | None] = None,
    # collocation_sizes should be:
    # [total_collocation_size] or
    # [interior_size, boundary_size, quadrature_size]
    collocation_sizes: Optional[Sequence[int] | None] = None,
    # repeat number of inner batch, for training the same batch with
    # {repeat} steps of different collocation points
    batch_repeat: Optional[int | None] = None,
) -> Generator[FeatureDict, None, None]:
    tfds_split = _to_tfds_split(split, split_percentage)
    ds, info = tfds.load(
        name,
        data_dir=tfds_dir,
        split=tfds_split,
        shuffle_files=True,
        with_info=True,
    )
    # tf.data options
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False
    ds = ds.with_options(options)

    if is_training:
        ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(
            buffer_size=info.splits[tfds_split].num_examples,
            reshuffle_each_iteration=True,
        )

        if batch_repeat:
            ds = repeat_batch(batch_sizes, batch_repeat)(ds)

    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    if is_training and collocation_sizes:
        rng = tf.random.Generator.from_seed(seed=0)
        collocation_axis_dict = (
            info.metadata["phase_feature_axis"],
            info.metadata["boundary_feature_axis"],
        )
        ds = ds.map(
            sample_collocation_coords(collocation_sizes, collocation_axis_dict, rng),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    yield from tfds.as_numpy(ds)


def _to_tfds_split(split: Split, split_percentage: str = ""):
    if split in (Split.TRAIN, Split.TRAIN_AND_VALID):
        return f"train[:{split_percentage}]"
    elif split in (Split.VALID, Split.TEST):
        return f"train[{split_percentage}:]"
    else:
        raise ValueError(f"Unknown split {split}")


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def sample_collocation_coords(
    batch,
    collocation_sizes: list[int],
    collocation_axis_dicts: list[dict],
    generator: Generator,
):
    """Sample phase points randomly and take collocation points.

    Args:
        featrues: batch to sample.
        collocation_sizes: number of collocation points.
        seed: random seed.

    Returns:
        sampled data.
    """

    phase_feature_axis = collocation_axis_dicts[0]
    num_phase_coords = tf.shape(batch["phase_coords"])[
        phase_feature_axis["phase_coords"]
    ]
    phase_coords_indices = generator.uniform(
        (collocation_sizes[0],),
        minval=0,
        maxval=num_phase_coords,
        dtype=tf.int32,
    )
    for k, axis in phase_feature_axis.items():
        if k in batch:
            batch[k] = tf.gather(batch[k], phase_coords_indices, axis=axis)

    if len(collocation_sizes) > 1:
        boundary_feature_axis = collocation_axis_dicts[1]
        num_boundary_coords = tf.shape(batch["boundary_coords"])[
            boundary_feature_axis["boundary_coords"]
        ]
        boundary_coords_indices = generator.uniform(
            (collocation_sizes[1],),
            minval=0,
            maxval=num_boundary_coords,
            dtype=tf.int32,
        )
        for k, axis in boundary_feature_axis.items():
            if k in batch:
                batch["sampled_" + k] = tf.gather(
                    batch[k], boundary_coords_indices, axis=axis
                )

    return batch


@curry1
def repeat_batch(
    ds: tf.data.Dataset, batch_sizes: int | Sequence[int], repeat: int = 1
) -> tf.data.Dataset:
    """Tiles the inner most batch dimension."""
    if repeat <= 1:
        return ds
    # Perform regular batching with reduced number of elements.
    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    # Repeat batch.
    repeat_fn = lambda x: tf.tile(  # noqa: E731
        x, multiples=[repeat] + [1] * (len(x.shape) - 1)
    )

    def repeat_inner_batch(example):
        return tf.nest.map_structure(repeat_fn, example)

    ds = ds.map(repeat_inner_batch, num_parallel_calls=tf.data.AUTOTUNE)
    # Unbatch.
    for _ in batch_sizes:
        ds = ds.unbatch()
    return ds
