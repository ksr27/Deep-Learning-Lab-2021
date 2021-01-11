import logging
import gin
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from input_pipeline.datasets import load
from models.architectures import vgg_like
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
    gin.parse_config_files_and_bindings(['/Users/lydiaschoenpflug/Documents/Master/WS20-21/DL-Lab/dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)

    trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


hyperopt = HyperOptSearch(metric="val_accuracy", mode="max")
config = {
    "prepare.clahe_flag": tune.choice([True, False]),
    "preprocess.he_flag": tune.choice([True, False]),
    #"Trainer.total_steps": tune.grid_search([4e3]),
    "vgg_like.base_filters": tune.choice([8, 16]),
    "vgg_like.n_blocks": tune.choice([3, 4, 6]),
    "vgg_like.dense_units": tune.choice([32, 64]),
    "vgg_like.dropout_rate": tune.choice([0.1, 0.2, 0.3, 0.4]),
    "apply_clahe.clip_limit": tune.uniform(0.0, 8.0)#,
    #"prepare.batch_size": tune.choice([16, 32])
}

analysis = tune.run(
    train_func, num_samples=2, resources_per_trial={'gpu': 0, 'cpu': 2},
    config=config,
    search_alg=hyperopt)

print("Best config for val accuracy: ", analysis.get_best_config(metric="val_accuracy",mode="max", scope="all"))
#print("Best config for eval accuracy: ", analysis.get_best_config(metric="eval_accuracy",mode="max", scope="all"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
