import dataclasses
from typing import Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

from . import mapping
from .tf import project_features
from .utils import mean_squared_loss_fn

FEATURES = project_features.FEATURES
VMAP_AXES = {k: 0 if True else None for k in FEATURES}


@dataclasses.dataclass
class ProjectName(hk.Module):
    config: ConfigDict
    name: Optional[str] = "project_name"

    def __call__(self, batch, is_training, compute_loss=False, compute_metrics=False):
        c = self.config
        gc = self.config.global_config
        ret = {}

        def project_op(x):
            x = Block1(c.model.block_1, gc)(x)
            out = Block2(c.model.block_2, gc)(x)
            return out

        low_memory = (
            None if is_training or hk.running_init() else gc.subcollocation_size
        )
        batched_project_op = hk.vmap(
            mapping.sharded_map(
                project_op, shard_size=low_memory, in_axes=(VMAP_AXES,)
            ),
            split_rng=(not hk.running_init()),
        )

        batched_inputs = {k: batch[k] for k in FEATURES}
        predictions = batched_project_op(batched_inputs)
        ret.update({"predictions": predictions})

        if compute_loss:
            labels = batch["labels"]
            loss_1 = mean_squared_loss_fn(predictions, labels)
            total_loss = gc.loss1_weights * loss_1
            ret["loss"] = {
                "mse": loss_1,
                "loss_1_rmspe": jnp.sqrt(loss_1 / jnp.mean(labels**2)),
            }
            ret["loss"].update({"total": total_loss})

        if compute_metrics:
            labels = batch["labels"]
            mse = mean_squared_loss_fn(predictions, labels, axis=-1)
            relative_mse = mse / jnp.mean(labels**2)
            ret.update({"metrics": {"mse": mse, "rmspe": relative_mse}})

        if compute_loss:
            return total_loss, ret

        return ret


@dataclasses.dataclass
class Block1(hk.Module):
    config: ConfigDict
    global_config: ConfigDict
    name: Optional[str] = "block_1"

    def __call__(self, batch, is_training):
        pass


@dataclasses.dataclass
class Block2(hk.Module):
    config: ConfigDict
    global_config: ConfigDict
    name: Optional[str] = "block_2"

    def __call__(self, batch, is_training):
        pass
