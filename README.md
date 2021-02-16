# Team15
- Lydia Schönpflug (st169955)
- Baran Can Gül (st168861)

## Project 1: Diabetic Retinopathy

### Project structure
Project1-Diabetic_Retinopathy/custom_tfds: contains "idrid" and a subset of "kaggle_dr" dataset in tfds form
Project1-Diabetic_Retinopathy/diabetic_retinopathy:
- *main.py* : Run the code with this file
- *train.py* : Contains trainer class for model training
- *tune.py* : For running hyperparameter optimization
- *configs/config.gin*: All configurations for model architecture, dataset loading, visualization etc. can be set here.
- *input_pipeline/datasets.py*: Load dataset and preprocess.
                */preprocessing.py*: Contains preprocessing functions.
                */img_processing.py*: Clahe, Ben Graham preprocessing functions
                */visualization.py*: Confusion matrix visualization
- *models/architectures.py*: Defines model architecture
        */layers.py*: Defines single vgg_block
- *evaluation/metrics.py*: Metric classes
            */evaluate.py*: Function for evaluating the model on ds_test
- *deep visualization/grad_cam.py*: Guided grad cam implementation
- *logs*: During train/ evaluation run tensorboard and grad cam images (for evaluation) will be logged here.
- *tf_ckpts*: During train/ evaluation run checkpoints will be stored here.
- *documentation*: Contains our poster and presentation
- *results/avg10*: Contains logs and checkpoints for 10 runs for each configuration (basis for our average 10 runs result)
         */best_runs*: Contains logs and checkpoints for the best run for each configuration
- *examples_images/btgraham*: Before and after images for Ben Graham image processing.
                 */clahe*: Before and after images for clahe
                */data_augmentation*: Before and after images for data augmentation
                 */grad_cam*: Before and after images for Grad Cam (for all three image processing options)
                 */img_evolution*: Shows one images progression through all preprocessing steps.

### How to run the code
As we generated custom tfds from "idrid" and a subset of "kaggle_dr" dataset you have to adjust the path to the custom_tfds folder
*"dl-lab-2020-team15/Project1-Diabetic_Retinopathy/custom_tfds"* for your server user account/ computer in the *config.gin* file.
So f.e. load.data_dir = '/home/RUS_CIP/st169955/dl-lab-2020-team15/Project1-Diabetic_Retinopathy/custom_tfds'

#### To run in train mode:
in *main.py*:
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

Now you can just run *"python main.py"*

#### To run in eval mode:
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

Evaluate function:
evaluate(model,
         checkpoint='./results/best_runs/clahe/tf_ckpts/ckpt-24', # add the wanted checkpoint path here (see below)
         ds_test=ds_test,
         ds_info=ds_info,
         visualize_flag=True) Set this flag to run grad cam on a batch of images (logged to ./logs/eval/-timestamp-/grad_cam)

Take a checkpoint from the following list to evaluate our model on:
1. No image processing:       *'./results/best_runs/no-processing/tf_ckpts/ckpt-45'*
   Please adjust the config gin: prepare.processing_mode = 'none'
2. ben graham img processing: *'./results/best_runs/ben-graham/tf_ckpts/ckpt-48'*
   Please adjust the config gin: prepare.processing_mode = 'btg' 
3. clahe img processing:      *'./results/best_runs/clahe/tf_ckpts/ckpt-24'*
   Please adjust the config gin: prepare.processing_mode = 'clahe'
4. No augmentation:           *'./results/best_runs/no-augmentation/tf_ckpts/ckpt-74'*
   Please adjust the config gin: prepare.processing_mode = 'none'
5. No balancing:              *'./results/best_runs/plain/tf_ckpts/ckpt-78'*
   Please adjust the config gin: prepare.processing_mode = 'none'

### Results
Best overall: 
-Balanced Train Accuracy: **94,78%**
-Balanced Validation Accuracy: **93,27%**
-Balanced Test Accuracy: **89,62%**

checkpoint: *'./results/best_runs/no-processing/tf_ckpts/ckpt-45'*

see */dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/results* and */dl-lab-2020-team15/Project1-Diabetic_Retinopathy/diabetic_retinopathy/documentation* for more detailed information on our results
