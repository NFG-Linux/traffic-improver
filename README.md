# traffic-improver
uses AI City Challenge 2022, Track 1 dataset to improve traffic light functionality and decrease peak traffic volumes

Not included is the data contained within the train validation and test folders (prevents license issues; academic use only)

each of those folders has one or many S0x folders, and within each S0x folder contains several c0XX folders. See below for info

There is a .avi file in each c0xx folder (not in github repo due to size) Obtain from the AI City Challenge 2022 Track 1 zip file:
http://www.aicitychallenge.org/2022-track1-download/

if you're reading this, the github repo was most likely cloned... if not, CLONE THE GITHUB REPO 1ST! then download the zip file and copy the contents held in the train validation and test folders into the corresponding gitrepo folders

The yolo_models directory should have the yolov8 model weight files (v4 files not needed); obtained from coursework (not included in repo)

Steps to set up project:
1.	Clone github repo
2.	Download and unzip the AI City Challenge Dataset (total size about 15Gb)
3.	Copy the train, validation, and test folders from dataset into github project folder (root level) – there are empty folders in place already, safe to overwrite
4.	From the CPV course data set, copy the yolo_models folder(the yolov4 items not needed) into github project folder (root level) – there is an empty folder in place already, safe to overwrite
5.	Normally, you would run the tracker_yolov8.py script to generate the prediction file, but this was done already to save time
6.	Run either or both of the commands below:
Background: The dataset comes with an eval.py script (in eval folder). The tracker_yolov8.py script needs to be ran first and is set up to generate the prediction file that feeds into the eval.py script as well as a whitener file that should increase cross-camera ID more accurate. It takes a long time to generate these files, so I kept the generated files in the github code (in for_eval folder).  The data is split by dstype and each file feeds into separate runs of the eval.py To run the eval.py script run the below command(s) from the root of the github project (2 will be present below, but either one will work):
python3 eval/eval.py eval/ground_truth_validation.txt for_eval/validation_prediction.txt --dstype validation
python3 eval/eval.py for_eval/ground_truth_train.txt for_eval/train_prediction.txt --dstype train
