## Project 1: Diabetic Retinopathy
### Project structure
```
   - custom_tfds                      : IDRiD (Indian Diabetic Retinopathy Image Dataset) in custom tfds format
   - train.py                         : Trainer class for model training
   - tune.py                          : Hyperparameter optimization
   - configs/config.gin               : Configurations for training and evaluation
   - input_pipeline                   : Dataset loader, preprocessing functions
   - models                           : Model architecture
   - evaluation                       : Metrics and evalautor class
   - deep visualization               : Guided grad cam implementation
   - documentation                    : Poster and presentation
   - results/avg10                    : Contains logs and checkpoints for 10 runs for each configuration (basis for our average 10 runs result)
           /best_runs                 : Contains logs and checkpoints for the best run for each configuration
```

### How to run the code

#### Training:
in *main.py*:
```python
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
```

#### To run in eval mode:
```python
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
```
Configurations in config.gin:
```
# Evaluation
evaluate.checkpoint = ' ' # checkpoint to evaluate for
evaluate.visualize_flag = True # whether to run grad cam or not

# Deep Visualization
grad_cam_wbp.num_of_batches = 1 # num of batches to run gradcam for
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

see [results-folder](Project1-Diabetic_Retinopathy/diabetic_retinopathy/results) and [poster](Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation/poster-team15.pdf) and [presentation](Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation/presentation-team15.pdf) for more detailed information on our results

## Project 2: Human Activity Recognition
### Project structure
```
Project2-Human_Activity_Recognition/self_recorded_ds           : contains csv file recorded dataset
Project2-Human_Activity_Recognition/human_activity_recognition : main project folder

   - train.py                         : Trainer class for model training
   - tune.py                          : Hyperparameter optimization
   - configs/config.gin               : Configurations
   - configs/config.gin               : Configurations for training and evaluation
   - input_pipeline                   : Dataset loader, preprocessing functions
   - models                           : Model architecture
   - evaluation                       : Metrics and evalautor class
   - documentation                    : Project paper
   - best_runs/basic                  : logs, checkpoint and train, val and test confusion matrices for basic s2s and s2l model
              /arch-opt               : logs, checkpoint and train, val and test confusion matrices after hyperparam opt+ adding attention
              /loss-opt               : logs, checkpoint and train, val and test confusion matrices for different loss configuration
```

### How to run the code

#### Training:
in *main.py*:
```python
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
```
Configuration in config.gin:
```
Trainer.log_cm = True # whether to save all confusion matrices from training to file
```

#### To run in eval mode:
```python
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
```

Configurations in config.gin:
```
Evaluator.checkpoint     =   # * add the wanted checkpoint path here (see below)*  
Evaluator.visualize_flag =   # Set this flag to visualize the model prediction (logged to ./logs/eval/-timestamp)
Evaluator.num_batches    = 1 # num of batches to visualize the output prediction for
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

see [paper](Project2-Human_Activity_Recognition/human_activity_recognition/documentation/paper.pdf) 
