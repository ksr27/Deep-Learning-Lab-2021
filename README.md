# Team15
- Lydia Schönpflug (st169955)
- Baran Can Gül (st168861)

## Project 1: Diabetic Retinopathy
### Project structure
```
Project1-Diabetic_Retinopathy/custom_tfds         : contains "idrid" and a subset of "kaggle_dr" dataset in tfds form
Project1-Diabetic_Retinopathy/diabetic_retinopathy: main project folder

   - main.py                          : Run the code with this file
   - train.py                         : Contains trainer class for model training
   - tune.py                          : For running hyperparameter optimization
   - configs/config.gin               : All configurations for model architecture, dataset loading, visualization etc. can be set here.
   - input_pipeline/datasets.py       : Load dataset and preprocess.
                   /preprocessing.py  : Contains preprocessing functions.
                   /img_processing.py : Clahe, Ben Graham preprocessing functions
                   /visualization.py  : Confusion matrix visualization
   - models/architectures.py          : Defines model architecture
           /layers.py                 : Defines single vgg_block
   - evaluation/metrics.py            : Metric classes
              /evaluate.py            : Function for evaluating the model on ds_test
   - deep visualization/grad_cam.py   : Guided grad cam implementation
   - logs                             : During train/ evaluation run tensorboard and grad cam images (for evaluation) will be logged here.
   - tf_ckpts                         : During train/ evaluation run checkpoints will be stored here.
   - documentation                    : Contains our poster and presentation
   - results/avg10                    : Contains logs and checkpoints for 10 runs for each configuration (basis for our average 10 runs result)
           /best_runs                 : Contains logs and checkpoints for the best run for each configuration
   - examples_images/btgraham         : Before and after images for Ben Graham image processing.
                    /clahe            : Before and after images for clahe
                    /data_augmentation: Before and after images for data augmentation
                    /grad_cam         : Before and after images for Grad Cam (for all three image processing options)
                    /img_evolution    : Shows one images progression through all preprocessing steps.
```

### How to run the code
As we generated custom tfds from "idrid" and a subset of "kaggle_dr" dataset you have to adjust the path to the custom_tfds folder
`"dl-lab-2020-team15/Project1-Diabetic_Retinopathy/custom_tfds"` for your server user account/ computer in the `config.gin` file.
So f.e. ```python load.data_dir = '/home/RUS_CIP/st169955/dl-lab-2020-team15/Project1-Diabetic_Retinopathy/custom_tfds' ```

#### To run in train mode:
in *main.py*:
```python
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
```

Now you can just run `"python main.py"`

#### To run in eval mode:
```python
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
```

Evaluate function:
```python
evaluate(model,
         checkpoint='./results/best_runs/clahe/tf_ckpts/ckpt-24', # add the wanted checkpoint path here (see below)
         ds_test=ds_test,
         ds_info=ds_info,
         visualize_flag=True) #Set this flag to run grad cam on a batch of images (logged to ./logs/eval/-timestamp-/grad_cam)
```

Take a checkpoint from the following list to evaluate our model on:
1. No image processing:       `'./results/best_runs/no-processing/tf_ckpts/ckpt-45'` <br />
   Please adjust the config.gin: prepare.processing_mode = 'none'
2. ben graham img processing: `'./results/best_runs/ben-graham/tf_ckpts/ckpt-48'` <br />
   Please adjust the config.gin: prepare.processing_mode = 'btg'
3. clahe img processing:      `'./results/best_runs/clahe/tf_ckpts/ckpt-24'` <br />
   Please adjust the config.gin: prepare.processing_mode = 'clahe'
4. No augmentation:           `'./results/best_runs/no-augmentation/tf_ckpts/ckpt-74'` <br />
   Please adjust the config.gin: prepare.processing_mode = 'none'
5. No balancing:              `'./results/best_runs/plain/tf_ckpts/ckpt-78'` <br />
   Please adjust the config.gin: prepare.processing_mode = 'none'

### Results
Best overall:
- Balanced Train Accuracy: **94,78%**
- Balanced Validation Accuracy: **93,27%**
- Balanced Test Accuracy: **89,62%**

checkpoint: `'./results/best_runs/no-processing/tf_ckpts/ckpt-45' `

see */dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/results* and */dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation* for more detailed information on our results

## Project 2: Human Activity Recognition
### Project structure
```
Project2-Human_Activity_Recognition/self_recorded_ds           : contains csv filf reclorded dataset
Project2-Human_Activity_Recognition/human_activity_recognition : main project folder

   - main.py                          : Run the code with this file
   - train.py                         : Contains trainer class for model training
   - tune.py                          : For running hyperparameter optimization
   - configs/config.gin               : All configurations for model architecture, dataset loading, visualization etc. can be set here.
   - input_pipeline/datasets.py       : Load dataset and preprocess.
                   /preprocessing.py  : Dataset loading, preprocessing and managing tfrecord files
                   /visualization.py  : Confusion matrix visualization
   - models/architectures.py          : Defines model architecture
   - evaluation/metrics.py            : Metric classes
              /evaluate.py            : Class evaluator for evaluating the model on ds_test
   - logs                             : During train/ evaluation run tensorboard and grad cam images (for evaluation) will be logged here.
   - tf_ckpts                         : During train/ evaluation run checkpoints will be stored here.
   - tfrecords/s2s                    : tfrecord files for s2s classifcation
              /s2l                    : tfrecord files for s2l classifcation
   - documentation                    : Contains our paper
   - best_runs/basic                  : logs, checkpoint and train, val and test confusion matrices for basic s2s and s2l model
              /arch-opt               : logs, checkpoint and train, val and test confusion matrices after hyperparam opt+ adding attention
              /loss-opt               : logs, checkpoint and train, val and test confusion matrices for different loss configuration
```

### How to run the code
To use the self-recorded dataset, please adjust the folder path for your server user account/ computer in the `config.gin`:
`"dl-lab-2020-team15/Project2-Human_Activity_Recognition/self_recorded_ds"`
So f.e. ```python load.data_dir = '/home/RUS_CIP/st169955/dl-lab-2020-team15/Project2-Human_Activity_Recognition/self_recorded_ds' ```

#### To run in train mode:
in *main.py*:
```python
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
```

Now you can just run `"python main.py"`

#### To run in eval mode:
```python
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
```
Now you can just run `"python main.py"`

Evaluator:
```python
evaluator = Evaluator(model,
                      './best_runs/basic/s2l/tf_ckpts/ckpt-77', # add the wanted checkpoint path here (see below)
                      ds_test,
                      ds_info,
                      visualize_flag=False) #Set this flag to visualize the model prediction (logged to ./logs/eval/-timestamp)
```

Take a checkpoint from the following list to evaluate our model on:
S2L checkpoints:      
1. basic s2l model: `'./best_runs/basic/s2l/tf_ckpts/ckpt-77'` <br />
   Please adjust the config.gin:
   ```
   hapt_params.mode = 's2l'
   lstm_arch.lstm_units = 128
   lstm_arch.lstm_layers = 1
   lstm_arch.dense_units = 128
   lstm_arch.attention = False
   ```
2. after hyperparameter opt: `'./best_runs/arch-opt/s2l/tf_ckpts/ckpt-65'` <br />
    Please adjust the config.gin:
    ```
    hapt_params.mode = 's2l'
    lstm_arch.lstm_units = 256
    lstm_arch.lstm_layers = 2
    lstm_arch.dense_units = 256
    lstm_arch.attention = False
    ```
3. hyperparameter opt+ attention: `'./best_runs/arch-opt/s2l-attention/tf_ckpts/ckpt-67'` <br />
    Please adjust the config.gin:
    ```
    hapt_params.mode = 's2l'
    lstm_arch.lstm_units = 256
    lstm_arch.lstm_layers = 2
    lstm_arch.dense_units = 256
    lstm_arch.attention = True
    ```
5. SCCE with weighting:          `'./best_runs/loss-opt/s2l/scce-weighting/tf_ckpts/ckpt-57'` <br />
   Please adjust the config.gin:
   ```
   hapt_params.mode = 's2l'
   lstm_arch.lstm_units = 256
   lstm_arch.lstm_layers = 2
   lstm_arch.dense_units = 256
   lstm_arch.attention = True
      ```
6. Focal loss:              `'./best_runs/loss-opt/s2l/focal-loss/tf_ckpts/ckpt-68'` <br />
    Please adjust the config.gin:
    ```
    hapt_params.mode = 's2l'
    lstm_arch.lstm_units = 256
    lstm_arch.lstm_layers = 2  
    lstm_arch.dense_units = 256
    lstm_arch.attention = True
       ```
7. Focal loss+ weighting:  `'./best_runs/loss-opt/s2l/focal-loss-weighting/tf_ckpts/ckpt-65'` <br />
    Please adjust the config.gin:
    ```
    hapt_params.mode = 's2l'
    lstm_arch.lstm_units = 256
    lstm_arch.lstm_layers = 2  
    lstm_arch.dense_units = 256
    lstm_arch.attention = True
       ```

S2S checkpoints:
1. basic s2s model: `'./best_runs/basic/s2s/tf_ckpts/ckpt-70'` <br />
  Please adjust the config.gin:
  ```
  hapt_params.mode = 's2s'
  lstm_arch.lstm_units = 128
  lstm_arch.lstm_layers = 1
  lstm_arch.dense_units = 128
  ```
2. after hyperparameter opt: `'./best_runs/arch-opt/s2s/tf_ckpts/ckpt-75'` <br />
   Please adjust the config.gin:
   ```
   hapt_params.mode = 's2s'
   lstm_arch.lstm_units = 256
   lstm_arch.lstm_layers = 2
   lstm_arch.dense_units = 256
   ```
3. SCCE with weighting          `'./best_runs/loss-opt/s2s/scce-weighting/tf_ckpts/ckpt-74'` <br />
  Please adjust the config.gin:
  ```
  hapt_params.mode = 's2s'
  lstm_arch.lstm_units = 256
  lstm_arch.lstm_layers = 2
  lstm_arch.dense_units = 256
     ```
4. Focal loss:              `'./best_runs/loss-opt/s2s/focal-loss/tf_ckpts/ckpt-69'` <br />
   Please adjust the config.gin:
   ```
   hapt_params.mode = 's2l'
   lstm_arch.lstm_units = 256
   lstm_arch.lstm_layers = 2
   lstm_arch.dense_units = 256
      ```
5. Focal loss+ weighting:  `'./best_runs/loss-opt/s2s/focal-loss-weighting/tf_ckpts/ckpt-67'` <br />
   Please adjust the config.gin:
   ```
   hapt_params.mode = 's2l'
   lstm_arch.lstm_units = 256
   lstm_arch.lstm_layers = 2
   lstm_arch.dense_units = 256
   ```
### Results

S2L
Best overall: scce+weighting
- Balanced Train Accuracy: **97,20%**
- Balanced Validation Accuracy: **90,29%**
- Balanced Test Accuracy: **92,30%**
- Balanced Accuracy on self_recorded_ds: **21,05%**

checkpoint: `'./best_runs/loss-opt/s2l/scce-weighting/tf_ckpts/ckpt-57'` <br />

S2S
Best overall: scce+weighting
- Balanced Train Accuracy: **94,16%**
- Balanced Validation Accuracy: **79,35%**
- Balanced Test Accuracy: **80,94%**
- Balanced Accuracy on self_recorded_ds: **22,82%**

checkpoint: `'./best_runs/loss-opt/s2s/scce-weighting/tf_ckpts/ckpt-74'` <br />

see */dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/results* and */dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation* for more detailed information on our results
