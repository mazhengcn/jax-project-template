import copy

import ml_collections


def model_config() -> ml_collections.ConfigDict:
    cfg = copy.deepcopy(CONFIG)
    return cfg


CONFIG = ml_collections.ConfigDict(
    {
        "data": {"is_normalization": True},
        "global_config": {
            "deterministic": True,
            "subcollocation_size": 128,
            "w_init": "glorot_uniform",
            "loss_weights": 5.0,
            "bc_loss_weights": 1.0,
        },
        "model": {"block_1": {}, "block_2": {}},
    }
)
