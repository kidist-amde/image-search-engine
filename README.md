# image-search-engine
 The main objective of this project is to create an algorithm able to match the images in a query set with the images in a much larger set called gallery.
##### Installing dependancies.
Create your connda enviroment 
```bash
conda create -n "enviroment-name" python=3.6
```

and install the dependancies.
```bash
pip install -r requirment.txt
```

##### Running feature extraction script.
Assume you have dataset folder in your current directory and in side this directory you have validation directory which hold query and gallery folder inside it. You can use the following command to excute the feature extraction script and to retrieve the top 10 similar images . 
```bash
python feature_extraction.py -q dataset/validation/query/4/ec50k_00040001.jpg -g dataset/validation/gallery/ -k 10
```
for more information you can use the following command
```bash 
python feature_extraction.py --help
```
##### cofigration 

Replace the file paths which specfied in the [config](config.py) script.

* `submission_data_path` : Replace with your test data set path
* `url` : Put the url of the server 
* `download_data_path `: Path to the downloaded dataset
* `split_data_path` : Path to the training and validation split of downloaded dataset
* `train_path`: Path to the classifier training dataset
* `val_path`: Path to the classifier validation dataset 
* `best_model_path`: Where to save classifier best model
* `best_state_path `: Where to save classifier best state 
 