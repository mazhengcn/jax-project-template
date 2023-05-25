"Utilities functions."
from collections.abc import Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(0.87962566103423978, dtype=np.float32)


def mean_squared_loss_fn(x, y, axis=None):
    return jnp.mean(jnp.square(x - y), axis=axis)


def apply_dropout(*, tensor, safe_key, rate, is_training):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
        keep_rate = 1.0 - rate
        keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
        return keep * tensor / keep_rate
    else:
        return tensor


def dropout_wrapper(
    module,
    input_act,
    kernel,
    safe_key,
    global_config,
    output_act=None,
    is_training=True,
    **kwargs,
):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act

    gc = global_config
    residual = module(input_act, kernel, is_training=is_training, **kwargs)
    dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

    residual = apply_dropout(
        tensor=residual,
        safe_key=safe_key,
        rate=dropout_rate,
        is_training=is_training,
    )

    new_act = output_act + residual

    return new_act


def get_initializer_scale(initializer_name, input_shape=()):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == "zeros":
        w_init = hk.initializers.Constant(0.0)
    elif initializer_name == "glorot_uniform":
        w_init = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )
    else:
        # fan-in scaling
        scale = 1.0
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == "relu":
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


def flat_params_to_haiku(params: Mapping[str, np.ndarray]) -> hk.Params:
    """Convert a dictionary of NumPy arrays to Haiku parameters."""
    hk_params = {}
    for path, array in params.items():
        scope, name = path.split("//")
        if scope not in hk_params:
            hk_params[scope] = {}
        hk_params[scope][name] = jnp.array(array)

    return hk_params
