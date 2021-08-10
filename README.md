# Brain Tumor Detection

The goal is to train a neural network to detect brain tumors in MRI images. The data is from this dataset: https://www.kaggle.com/ahmedhamada0/brain-tumor-detection.

## Execution Instructions

### Requirements
We ran the code on Python 3.8 and Tensorflow 2.3.1. Model training was done on GPU.

Other requirements can be installed using this command:
```bash
pip3 install -r requirements.txt
```

### Download the Data
 1. Download the data from this link: https://www.kaggle.com/ahmedhamada0/brain-tumor-detection/download
 2. Extract the data to the *raw_data* folder
 3. The *raw_data* folder should contain a *yes* folder and a *no* folder

 ### Data Formatting
 The data needs to be formatted before running the experiments. The *preprocessing.ipynb* script can be used to do this automatically.

 At the end of data formatting, there should be two new directories: *train_data* and *test_data*. These folders should each have a *yes* folder and a *no* folder.
 
 ### Run the Experiments
 Individual experiments can be run through the *experiment_X.ipynb* notebooks. These experiments should populate a *models* folder with the trained model on completion.

 ### Evaluate the Experiments
The trained models can be evaluated against the test set using the *model_evaluation.ipynb* notebook. This script will evaluate all of the models within the *models* directory.
