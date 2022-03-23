# One shot Video object segmentation

## Data

### YouTube-VOS

Download the YouTube-VOS dataset from their [website](https://youtube-vos.org/). 
Create a folder named ```databases```in the parent folder of the root directory of this project and put there the database in a folder named ```YOUTUBE```.
### DAVIS 2017

Download the DAVIS 2017 dataset from their [website](https://davischallenge.org/davis2017/code.html) at 480p resolution. Create a folder named ```databases```in the parent folder of the root directory of this project and put there the database in a folder named ```DAVIS2017```. The root directory (```rvos```folder) and the ```databases``` folder should be in the same directory.

## Training
To model has three settings, each one can be configured easly through the running arguments in the main of train_previous_mask.py:
1. Intial setting: set args.use_flip to 0 and args.use_GS_hidden to 0
2. Extended input for decoder - set args.use_flip to 0 and args.use_GS_hidden to 1
3. Loss metric - set args.use_flip to 1 and args.use_GS_hidden to 0

Configure the training to the desired model, and run train_previous_mask.py to train the model

## Evaluation
To evaluate the results of a certain model, define the directory of the model in the main of eval_previous_mask.py and run the script. 
The script saves the masks the model evaluated on DAVIS data set, and prints the region similarity (J) and contour accuracy (F) of the model. 
