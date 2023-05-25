import ml_collections
import tensorflow_datasets as tfds
from jaxline import base_config

from .model.config import model_config


def get_config():
    config = base_config.get_base_config()

    num_epochs = 5000

    dataset_config = ml_collections.ConfigDict(
        dict(name="project_name", data_dir="data/tfds", split_percentage="80%")
    )

    dataset_builder = tfds.builder(
        dataset_config.name, data_dir=dataset_config.data_dir
    )
    dataset_config.num_train_examples = dataset_builder.info.splits[
        f"train[:{dataset_config.split_percentage}]"
    ].num_examples

    model = model_config()
    model.data.normalization_dict = dataset_builder.info.metadata["normalization"]

    training_config = ml_collections.ConfigDict(
        dict(
            num_epochs=num_epochs,
            batch_size=8,
            collocation_sizes=[128],
            batch_repeat=1,
            accum_grads_steps=1,
        )
    )

    def steps_from_epochs(num_epochs):
        return max(
            int(
                training_config.batch_repeat
                * num_epochs
                * dataset_config.num_train_examples
                // training_config.batch_size
            ),
            1,
        )

    config.training_steps = steps_from_epochs(num_epochs)

    config.experiment_kwargs = ml_collections.ConfigDict(
        dict(
            config=dict(
                dataset=dataset_config,
                training=training_config,
                optimizer=dict(
                    base_lr=1e-3,
                    scale_by_batch=False,
                    schedule_type="exponential",
                    decay_kwargs=dict(
                        transition_steps=steps_from_epochs(100), decay_rate=0.96
                    ),
                    optimizer="adam",
                    adam_kwargs=dict(),
                ),
                evaluation=dict(batch_size=4),
                model=model,
            )
        )
    )

    config.interval_type = "steps"
    config.save_checkpoint_interval = steps_from_epochs(10)
    config.log_tensors_interval = steps_from_epochs(1)
    config.log_train_data_interval = steps_from_epochs(1)
    # When True, the eval job immediately loads a checkpoint
    # runs evaluate() once, then terminates.
    config.one_off_evaluate = False
    # Seed for the RNGs (default is 42).
    config.random_seed = 42
    config.checkpoint_dir = ""
    config.restore_dir = ""

    return config
