import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from deep_visualization.grad_cam import grad_cam_wbp

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    #model.build(input_shape=(8, 256, 256, 3))
    model.summary()
    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        grad_cam_wbp(model,'./tf_ckpts/', "conv2d_2", ds_train, 1, 256, 256)
        evaluate(model,
                 './tf_ckpts/',
                 ds_test,
                 ds_info,
                 run_paths)

if __name__ == "__main__":
    app.run(main)