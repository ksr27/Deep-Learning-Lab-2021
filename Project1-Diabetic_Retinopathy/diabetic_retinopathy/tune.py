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
    #bindings = []
    #for key, value in config.items():
       # bindings.append(f'{key}={value}')
    # generate folder structures
    # set loggers
   # utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
  #  for key, value in config.items():
   #     gin.bind_parameter('vgg_like.'+key, value)

    #utils_params.save_config(run_paths['path_gin'], gin.config_str())
    #run_paths = utils_params.gen_run_folder()  # ('_'.join(bindings))

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()


    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes,
                     base_filter=config['base_filter'],dense_unit=config['dense_unit'],dropout_rate=config['dropout_rate'])

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(),
                               net=model, iterator=iter(ds_train))
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
    for _ in trainer.train(False):
        continue
    print(run_paths)
    val_acc= evaluate(model,manager.latest_checkpoint,ds_val,ds_info,run_paths)

    return val_acc

base_filters=[8, 16]
dense_units=[32, 64]
dropout_rates =[0.1, 0.15, 0.2, 0.25]
run_paths = utils_params.gen_run_folder()  # ('_'.join(bindings))
utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
gin.parse_config_files_and_bindings(['configs/config.gin'], [])


for base_filter in base_filters:
    for dense_unit in dense_units:
        for dropout_rate in dropout_rates:
            dict={'base_filter':base_filter,'dense_unit':dense_unit, 'dropout_rate':dropout_rate}
            val_acc = train_func(dict,run_paths)

            logging.info(f'validation accuracy={val_acc} for base filter={base_filter}; dense unit={dense_unit}; '
                         f'dropout rate={dropout_rate}')

'''
analysis = tune.run(
    train_func, num_samples=2, resources_per_trial={'cpu': 4},
    config={
        "Trainer.total_steps": tune.grid_search([1e4]),
        "vgg_like.base_filter": tune.choice([8]),#, 16]),
        "vgg_like.n_blocks": tune.choice([2]),#, 3, 4, 5]),
        "vgg_like.dense_unit": tune.choice([32]),#, 64]),
        "vgg_like.dropout_rate": tune.choice([0.1]),
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
'''