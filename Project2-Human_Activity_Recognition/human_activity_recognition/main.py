import gin
import logging
from absl import app, flags

from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import lstm_arch

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
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    model = lstm_arch(input_shape=(ds_info['window_length'], ds_info['num_features']), n_classes=ds_info['num_classes'],
                      mode=ds_info['mode'])
    model.summary()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info)
        for _ in trainer.train():
            continue
    else:
        evaluator = Evaluator(model, './best_runs/basic/s2l/tf_ckpts/ckpt-77', ds_test, ds_info,
                              visualize_flag=False)
        # S2L checkpoints:
        # './best_runs/basic/s2s/tf_ckpts/ckpt-70'
        # './best_runs/basic/s2s/tf_ckpts/ckpt-75'
        # './best_runs/loss-opt/s2s/scce-weighting/tf_ckpts/ckpt-74'
        # './best_runs/loss-opt/s2s/focal-loss/tf_ckpts/ckpt-69'
        # './best_runs/loss-opt/s2s/focal-loss-weighting/tf_ckpts/ckpt-67'

        # S2L checkpoints:
        # './best_runs/basic/s2l/tf_ckpts/ckpt-77'
        # './best_runs/arch-opt/s2l/tf_ckpts/ckpt-65'
        # './best_runs/arch-opt/s2l-attention/tf_ckpts/ckpt-67'
        # './best_runs/loss-opt/s2l/scce-weighting/tf_ckpts/ckpt-57'
        # './best_runs/loss-opt/s2l/focal-loss/tf_ckpts/ckpt-68'
        # './best_runs/loss-opt/s2l/focal-loss-weighting/tf_ckpts/ckpt-65'
        evaluator.evaluate()


if __name__ == "__main__":
    app.run(main)
