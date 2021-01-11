import logging
import gin
from ray import tune
import tensorflow as tf
from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc
from evaluation.eval import evaluate

def train_func(config,run_paths):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
       bindings.append(f'{key}={value}')
    # generate folder structures
    # set loggers
   # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['~/configs/config.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())


    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)

    trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


analysis = tune.run(
    train_func, num_samples=2, resources_per_trial={'cpu': 4},
    config={
      #  "prepare.clahe_flag": tune.choice([True, False]),
      #  "preprocess.he_flag": tune.choice([True, False]),
        "vgg_like.base_filters": tune.choice([8,16]),
        "vgg_like.n_blocks": tune.choice([4, 6]),
        "vgg_like.dense_units": tune.choice([64, 128]),
        "vgg_like.dropout_rate": tune.uniform(0.1, 0.4),
        "balance_ds.aug_perc": tune.uniform(0.0,6.0),
      #  "apply_clahe.clip_limit": tune.uniform(0.0,6.0),
        "prepare.batch_size": tune.choice([16,32])
    })

print("Best config for val accuracy: ", analysis.get_best_config(metric="val_accuracy",mode="max", scope="all"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
