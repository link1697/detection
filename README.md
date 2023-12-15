# Pedestrian Detection

## Dependencies

* OpenCV
* scikit-image ```pip install scikit-image==0.14.1```
* scikkit-learn ```pip install scikit-learn==0.20.2```

## Running Detection

To test on images, simply run, `python3 detectmulti.py -i <path to image>`

For example, `python3 detectmulti.py -i sample_images/pedestrian.jpg`


For more options run, `python3 detectmulti.py -h`


## Training
This code is meant for Training a Pedestrian Detector using [INRIA Person Dataset] 
The above dataset can be downloaded from [here](https://drive.google.com/file/d/14GD_pBpBsprPiZlkmtXN_y5K72To16if/view?usp=sharing).
Download, Uncompress and place it in the root of this repository.

Then just run:
```
sudo sh fixpng.sh # To fix the broken png files in this dataset.
```
**Note:** *The above step is necessary to fix the broken pngs in the dataset, without running this the classifer won't be correctly trained.*

To train just run:
```
python3 train.py --pos <path to positive images> --neg <path to negative images>
```
For INRIA dataset, this would be probably,
```
python3 train.py --pos INRIAPerson/train_64x128_H96/pos --neg INRIAPerson/train_64x128_H96/neg
```

After training, two new files would be created, namely, `person.pkl` and `person_final.pkl`. The former is the pre eliminary detector and the latter is the improved (hard negatively mined) detector.

### Note on Training:
Training can use high amounts of memory, so be sure to have a swap space in case of RAM overflows. Also, memory consumption can be reduced by decreasing the maximum number of hard negative windows to be mined. This is defined by the `MAX_HARD_NEGATIVES` global variable in `train.py`.


## Testing
After successful training just run:
```
python3 test.py --pos <path to positive images> --neg <path to negative images>
```
For INRIA dataset, this would be probably,
```
python3 test.py --pos INRIAPerson/test_64x128_H96/pos --neg INRIAPerson/test_64x128_H96/neg
```

This would print `True Positives`, `True Negatives`, `False Positives`, `False Negatives`, `Precision`, `Recall` and `F1 Score`.
# detection
sudo apt-get install libgtk2.0-dev pkg-config
