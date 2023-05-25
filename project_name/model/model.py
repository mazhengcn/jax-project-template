from collections.abc import Mapping
from typing import Any, Optional

import haiku as hk
import jax
import ml_collections
from absl import logging

from . import features, modules


class RunModel:
    """Container for JAX model."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        params: Optional[Mapping[str, Mapping[str, jax.Array]]] = None,
        multi_devices: bool = False,
    ):
        self.config = config
        self.params = params
        self.multi_devices = multi_devices

        def _forward_fn(batch):
            model = modules.ProjectName(self.config)
            return model(
                batch,
                is_training=False,
                compute_loss=False,
                compute_metrics=False,
            )

        self.init = jax.jit(hk.transform(_forward_fn).init)

        if multi_devices:
            if self.multi_devices:
                # Replicate parameters across devices.
                self.params = jax.device_put_replicated(
                    self.params, jax.local_devices()
                )
            vmap_axes = jax.tree_map(lambda x: x + 1 if not x else x, modules.VMAP_AXES)
            self.apply = jax.pmap(
                hk.transform(_forward_fn).apply,
                in_axes=(0, None, vmap_axes),
                out_axes=1,
                axis_name="batch",
            )
        else:
            self.apply = jax.jit(hk.transform(_forward_fn).apply)

    def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
        """Initializes the model parameters."""

        if not self.params:
            # Init params randomly.
            rng = jax.random.PRNGKey(random_seed)
            self.params = hk.data_structures.to_mutable_dict(self.init(rng, feat))
            logging.warning("Initialized parameters randomly")

            if self.multi_devices:
                # Replicate parameters across devices.
                self.params = jax.device_put_replicated(
                    self.params, jax.local_devices()
                )

    def process_features(
        self, raw_features: features.FeatureDict
    ) -> features.FeatureDict:
        """Processes features to prepare for feeding them into the model."""

        if self.multi_devices:
            return features.np_data_to_features(raw_features, jax.local_device_count())
        else:
            return features.np_data_to_features(raw_features)

    def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
        self.init_params(feat)
        logging.info(
            "Running eval_shape with shape(feat) = %s",
            jax.tree_map(lambda x: x.shape, feat),
        )
        shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
        logging.info("Output shape was %s", shape)
        return shape

    def predict(
        self, feat: features.FeatureDict, random_seed: int = 0
    ) -> Mapping[str, Any]:
        """Makes a prediction by inferencing the model on the provided
        features.
        """
        self.init_params(feat)
        logging.info(
            "Running predict with shape(feat) = %s",
            jax.tree_map(lambda x: x.shape, feat),
        )
        result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)

        jax.tree_map(lambda x: x.block_until_ready(), result)

        logging.info(
            "Output shape was %s",
            jax.tree_map(lambda x: x.shape, result),
        )
        return result
