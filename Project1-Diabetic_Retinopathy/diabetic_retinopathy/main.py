import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info, ds_train_size, batch_size = datasets.load()

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=2)#ds_info.features["label"].num_classes)
    model.summary()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, ds_train_size,batch_size, run_paths)
        for _ in trainer.train():
            continue
    else:
        evaluate(model,
                 '/Users/lydiaschoenpflug/Documents/Master/WS20-21/DL-Lab/gpu_ray_results/best_runs/bt_graham_yes_no/tf_ckpts/20210119-142425',
                 ds_test,
                 True) #visualize Flag

if __name__ == "__main__":
    app.run(main)
