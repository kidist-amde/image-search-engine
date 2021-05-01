# image-search-engine
 The main objective of this project is to create an algorithm able to match the images in a query set with the images in a much larger set called gallery.
##### Running feature extraction script.
Assume you have dataset folder in your current directory and in side this directory you have validation directory which hold query and gallery folder inside it. You can use the followinf command to excute the feature extraction script. 
```bash
python feature_extraction.py -q dataset/validation/query/4/ec50k_00040001.jpg -g dataset/validation/gallery/
```
for more information you can use the following command
```bash 
python feature_extraction.py --help
```