import json
import os
import pathlib
import time
from typing import Any

import dill
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from absl import app, flags, logging
from matplotlib.colors import ListedColormap

from project_name.data import pipeline
from project_name.model import model
from project_name.model.utils import flat_params_to_haiku

logging.set_verbosity(logging.INFO)


flags.DEFINE_string("data_dir", None, "Path to directory containing the data.")
flags.DEFINE_list("data_filenames", None, "List of data filenames.")
flags.DEFINE_string("model_dir", None, "Path to directory containing the model.")
flags.DEFINE_string(
    "output_dir",
    None,
    "Path to output directory. If not specified, a directory will be created "
    "in the system's temporary directory.",
)
flags.DEFINE_bool("benchmark", True, "If True, benchmark the model.")


FLAGS = flags.FLAGS


def rmse_fn(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2) / np.mean(target**2))


def get_normalization_ratio(psi_range, boundary_range):
    psi_range = float(psi_range.split(" ")[-1])
    boundary_range = float(boundary_range.split(" ")[-1])
    return psi_range / boundary_range


def _jnp_to_np(output: dict[str, Any]) -> dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def plot_phi(r, phi_pre, phi_label, save_path):
    fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    fig.subplots_adjust(hspace=0.3)
    axs = _axs.flatten()

    viridis = matplotlib.colormaps["viridis"](np.linspace(0, 1.2, 128))
    cs_1 = axs[0].contourf(
        r[..., 0], r[..., 1], phi_label, cmap=ListedColormap(viridis)
    )
    axs[0].set_title(r"Exact $f(x,v)$", fontsize=20)
    axs[0].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_1)
    cbar.ax.tick_params(labelsize=16)

    cs_2 = axs[1].contourf(r[..., 0], r[..., 1], phi_pre, cmap=ListedColormap(viridis))
    axs[1].set_title(r"Predict $f(x,v)$", fontsize=20)
    axs[1].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_2)
    cbar.ax.tick_params(labelsize=16)

    cs_3 = axs[2].contourf(
        r[..., 0],
        r[..., 1],
        abs(phi_pre - phi_label),
        cmap=ListedColormap(viridis),
    )
    axs[2].set_title(r"Absolute error", fontsize=20)
    axs[2].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_3)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(save_path)


def predict_radiative_transfer(
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    model_runner: model.RunModel,
    benchmark: bool,
    normalization_ratio,
    random_seed: int,
):
    # Get features.
    raw_feature_dict = data_pipeline.process()
    num_examples = raw_feature_dict["shape"]["num_examples"]

    logging.info("Predicting %d examples sequentially", num_examples)

    output_dir_base = pathlib.Path(output_dir_base)
    if not output_dir_base.exists():
        output_dir_base.mkdir(parents=True)

    for i in range(num_examples):
        timings = {}

        logging.info("Predicting example %d/%d", i + 1, num_examples)

        output_dir = output_dir_base / f"example_{i}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        feature_dict = {
            "functions": jax.tree_map(
                lambda x: x[i : i + 1], raw_feature_dict["functions"]
            ),
            "grid": raw_feature_dict["grid"],
            "shape": raw_feature_dict["shape"],
        }

        # Write out features as a pickled dictionary.
        features_output_path = output_dir / "features.dill"
        with open(features_output_path, "wb") as f:
            dill.dump(feature_dict, f)

        # Run the model.
        logging.info("Running model...")
        t_0 = time.time()
        processed_feature_dict = model_runner.process_features(feature_dict)
        timings["process_features"] = time.time() - t_0

        t_0 = time.time()
        prediction_result = model_runner.predict(
            processed_feature_dict, random_seed=random_seed
        )
        t_diff = time.time() - t_0
        timings["predict_and_compile"] = t_diff

        if i == 0:
            logging.info(
                "Total JAX model predict time "
                "(includes compilation time, see --benchmark): %.1fs",
                t_diff,
            )
            if benchmark:
                t_0 = time.time()
                model_runner.predict(processed_feature_dict, random_seed=random_seed)
                t_diff = time.time() - t_0
                timings["predict_benchmark"] = t_diff
                logging.info(
                    "Total JAX model predict time "
                    "(excludes compilation time): %.1fs",
                    t_diff,
                )
        else:
            logging.info("Total JAX model predict time: %.1fs", t_diff)

        label_shape = feature_dict["functions"]["labels"].shape
        t_0 = time.time()
        predictions = (
            prediction_result["predictions"]
            .reshape(1, -1)  # reshape multi_devices to single device
            .reshape(label_shape)
        )
        if normalization_ratio:
            predictions = predictions * normalization_ratio

        t_diff = time.time() - t_0
        timings["compute_predictions"] = t_diff

        prediction_result.update({"predictions": predictions})

        # Remove jax dependency from results.
        np_prediction_result = _jnp_to_np(dict(prediction_result))
        # Save the model outputs.
        result_output_path = output_dir / "result.dill"
        with open(result_output_path, "wb") as f:
            dill.dump(np_prediction_result, f)

        # Compute metrics.
        metrics = {}
        labels = feature_dict["functions"]["labels"]
        rmse = rmse_fn(predictions, labels)
        metrics.update({"rmse": str(rmse)})
        logging.info("RMSE: %f", rmse)

        metrics_output_path = output_dir / "metrics.json"
        with open(metrics_output_path, "w") as f:
            f.write(json.dumps(metrics, indent=4))

        timings_output_path = output_dir / "timings.json"
        with open(timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))

        figure_save_path = output_dir / "plot.png"
        plot_phi(
            feature_dict["grid"]["position_coords"].reshape(*label_shape[1:-1], -1),
            predictions[0],
            labels[0],
            figure_save_path,
        )


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    data_pipeline = pipeline.DataPipeline(FLAGS.data_dir, FLAGS.data_filenames)
    logging.info("Data pipeline created from %s", FLAGS.data_dir)

    model_config_path = os.path.join(FLAGS.model_dir, "model.json")
    logging.info("Loading model config from %s", model_config_path)
    with open(model_config_path) as f:
        str = f.read()
        model_config = ml_collections.ConfigDict(json.loads(str))
    logging.info("Model config:\n%s", model_config)

    params_path = os.path.join(FLAGS.model_dir, "params.npz")
    np_params = np.load(params_path, allow_pickle=True)
    params = flat_params_to_haiku(np_params)
    logging.info("Model params loaded from %s", params_path)

    model_runner = model.RunModel(model_config, params, multi_devices=True)
    if model_config.data.is_normalization:
        normalization_dict = model_config.data.normalization_dict
        normalization_ratio = get_normalization_ratio(
            normalization_dict["psi_range"],
            normalization_dict["boundary_range"],
        )
    else:
        normalization_ratio = None

    logging.info("Running prediction...")
    predict_radiative_transfer(
        FLAGS.output_dir,
        data_pipeline,
        model_runner,
        FLAGS.benchmark,
        normalization_ratio,
        0,
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["data_dir", "data_filenames", "output_dir", "model_dir"]
    )
    app.run(main)
