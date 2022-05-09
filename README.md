## Project 1: Diabetic Retinopathy

### Abstract
Diabetic retinopathy (DR) is a side effect of diabetes and mainly detected through a time-consuming examination of retina images by trained personal. An automated detection could support doctors in recognizing and categorizing stages of DR, thus fastening the diagnosis and treatment process.
A CNN-based model was trained on the IDRiD dataset (Indian Diabetic Retinopathy Image Dataset) to classify images into referable (RDR) and non-referable diabetic retinopathy (NRDR.
Model performance was evaluated for three image preprocessing configurations. (1) Contrast limited adaptive histogram equalization (clahe), (2) preprocessing according to Ben Graham's approach int the Diabetic Retinopathy Kaggle challenge and (3) no additional preprocessing. The best overall accuracy was for no image processing. This might be due to the small size of the dataset and rather high quality of the dataset images, which proves further processing steps to be unnecessary. Improvements could however be achieved through balancing and augmenting the dataset, leading to a +7% increase for validation and test accuracy.

See [results-folder](Project1-Diabetic_Retinopathy/diabetic_retinopathy/results) and [poster](Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation/poster-team15.pdf) and [presentation](Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation/presentation-team15.pdf) for more detailed information on our results

### Results
Best overall:
- Balanced Train Accuracy: **94,78%**
- Balanced Validation Accuracy: **93,27%**
- Balanced Test Accuracy: **89,62%**

### Run code
`python main.py` add `--train` for training, else evaluation is run.

### Configurations:
```
# Evaluation
evaluate.checkpoint = './results/best_runs/no-processing/tf_ckpts/ckpt-45' # best perormin checkpoint
evaluate.visualize_flag = True # Grad cam deep visualization

# Deep Visualization
grad_cam_wbp.num_of_batches = 1 # Amount of visualization batches
```

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

## Project 2: Human Activity Recognition

### Abstract
Deep Learning approaches to Human Activity Recognition (HAR) are an active field of research with applications such as elderly and youth care, daily life monitoring or assisting Industry Manufacturing. Based on the Human Activities and Postural Transitions (HAPT) Dataset we developed a RNN-based classifier for both sequence-to-sequence (S2S) and sequence-to-label (S2L) classification. The models achieved 94.9% train, 78.81% validation and 94.0% test balanced accuracy for S2S- and 99.0% train, 84.24% validation and 88.43% test balanced accuracy for S2L-classification. As the HAPT dataset shows a great imbalance between basic activities and transition activties (91.38% to 8.62%) we proposed loss weighting and focal loss as ways to overcome this challenge. The overall best performance was achieved by using categorical cross entropy loss with loss weighting, resulting in 94.16% train, 79.35% validation and 80.94% test balanced accuracy for S2S- and 97.2% train, 90.3% validation and 92.3% test balanced accuracy for S2L-classification. Moreover we evaluated our model on a self-recorded dataset, the best results on this dataset was achieved by the S2S-model using focal loss with 30.56%.

For a detailed documentation see [paper](Project2-Human_Activity_Recognition/human_activity_recognition/documentation/paper.pdf).

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

### Run code
`python main.py` add `--train` for training, else evaluation is run.

### Configurations:
```
Evaluator.checkpoint     =   # checkpoint path
Evaluator.visualize_flag =   # Visualize model prediction
Evaluator.num_batches    = 1 # Amount of batches to visualize
```

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
