import collections.abc

import jax
import jax.numpy as jnp
import numpy as np


def accumulate_gradient(grad_fn, params, batch, batch_size, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""

    if accum_steps and accum_steps > 1:
        assert (
            batch_size % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {batch_size}"
        step_size = batch_size // accum_steps

        def dynamic_slice_feat(feat_dict, i, step_size):
            def slice_fn(x):
                return jax.lax.dynamic_slice(
                    x, (i,) + (0,) * (x.ndim - 1), (step_size,) + x.shape[1:]
                )

            return jax.tree_map(slice_fn, feat_dict)

        def acc_grad_and_loss(i, l_and_state):
            sliced_batch = dynamic_slice_feat(batch, i * step_size, step_size)
            grads_i, (scalars_i, state_i) = grad_fn(params, sliced_batch)
            grads, (scalars, state) = l_and_state
            return jax.tree_map(lambda x, y: x + y, grads, grads_i), (
                jax.tree_map(lambda x, y: x + y, scalars, scalars_i),
                state_i,
            )

        grads_shape_dtype = jax.eval_shape(
            grad_fn, params, dynamic_slice_feat(batch, 0, step_size)
        )
        l_and_state_0 = jax.tree_map(
            lambda sd: jnp.zeros(sd.shape, sd.dtype), grads_shape_dtype
        )
        grads, (scalars, state) = jax.lax.fori_loop(
            0, accum_steps, acc_grad_and_loss, l_and_state_0
        )
        return jax.tree_map(lambda x: x / accum_steps, (grads, (scalars, state)))
    else:
        return grad_fn(params, batch)


def hk_to_flat_dict(d, parent_key="", sep="//"):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
        path = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.Mapping):
            items.extend(hk_to_flat_dict(v, path, sep=sep).items())
        else:
            items.append((path, v))

    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
        items.append((parent_key, {}))

    return dict(items)


def flat_dict_to_hk(flat_dict):
    """Convert a dictionary of NumPy arrays to Haiku parameters."""
    hk_data = {}
    for path, array in flat_dict.items():
        scope, name = path.split("/")
        if scope not in hk_data:
            hk_data[scope] = {}
        hk_data[scope][name] = np.asarray(array)

    return hk_data
