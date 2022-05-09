import logging

import gin
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from input_pipeline.datasets import load
from models.architectures import lstm_arch
from train import Trainer
from utils import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['./configs/config.gin'], bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = lstm_arch(input_shape=(ds_info['window_length'], ds_info['num_features']), n_classes=ds_info['num_classes'],
                      mode=ds_info['mode'])

    trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


hyperopt = HyperOptSearch(metric="val_accuracy", mode="max")
config = {
    "lstm_arch.lstm_units": tune.choice([64, 128, 256]),
    "lstm_arch.lstm_layers": tune.choice([1, 2]),
    "lstm_arch.dense_units": tune.choice([64, 128, 256]),
    "lstm_arch.dropout_rate": tune.uniform(0.0, 0.5),
    "prepare.batch_size": tune.choice([16, 32, 64])
}

analysis = tune.run(
    train_func, num_samples=25, resources_per_trial={'gpu': 1, 'cpu': 10},
    config=config,
    search_alg=hyperopt)

print("Best config for val accuracy: ", analysis.get_best_config(metric="val_accuracy", mode="max", scope="all"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
