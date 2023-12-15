import sys
import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
import joblib
from skimage.feature import hog
from sklearn.utils import shuffle
import argparse
import random
from sklearn.decomposition import PCA
import joblib
MAX_HARD_NEGATIVES = 20000
PCA_N_COMPONENTS = 0.6
parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--pos', help='Path to directory containing Positive Images')
parser.add_argument('--neg', help='Path to directory containing Negative images')
# python3 train.py --pos INRIAPerson/train_64x128_H96/pos --neg INRIAPerson/train_64x128_H96/neg
args = parser.parse_args()
pos_img_dir = 'INRIAPerson/train_64x128_H96/pos'
neg_img_dir = 'INRIAPerson/train_64x128_H96/neg'

def crop_centre(img):
    h, w, _ = img.shape
    l = (w - 64) // 2
    t = (h - 128) // 2

    crop = img[t:t+128, l:l+64]
    return crop

def ten_random_windows(img):
    h, w = img.shape
    if h < 128 or w < 64:
        return []

    h = h - 128
    w = w - 64

    windows = []

    for i in range(10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        windows.append(img[y:y+128, x:x+64])

    return windows

def read_filenames():

    f_pos = []
    f_neg = []

    mypath_pos = pos_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_pos):
        f_pos.extend(filenames)
        break

    mypath_neg = neg_img_dir
    for (dirpath, dirnames, filenames) in os.walk(mypath_neg):
        f_neg.extend(filenames)
        break

    return f_pos, f_neg

def read_images(pos_files, neg_files):

    X = []
    Y = []

    pos_count = 0

    for img_file in pos_files:
        print(os.path.join(pos_img_dir, img_file))
        img = cv2.imread(os.path.join(pos_img_dir, img_file))

        cropped = crop_centre(img)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
        pos_count += 1

        X.append(features)
        Y.append(1)

    neg_count = 0

    for img_file in neg_files:
        print(os.path.join(neg_img_dir, img_file))
        img = cv2.imread(os.path.join(neg_img_dir, img_file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        windows = ten_random_windows(gray_img)

        for win in windows:
            features = hog(win, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            neg_count += 1
            X.append(features)
            Y.append(0)

    return X, Y, pos_count, neg_count

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0] - window_size[1], step_size[1]):
        for x in range(0, image.shape[1] - window_size[0], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def hard_negative_mine(f_neg, winSize, winStride):

    hard_negatives = []
    hard_negative_labels = []

    count = 0
    num = 0
    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            features = hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            # print('asf')
            features_pca = pca.transform(features.reshape(1, -1))
            # print(features_pca[0].shape)
            if (clf1.predict(features_pca) == 1):
                hard_negatives.append(features_pca.flatten())
                hard_negative_labels.append(0)
                count = count + 1

            if (count == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)

        num = num + 1

        sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %")

        sys.stdout.flush()

    return np.array(hard_negatives), np.array(hard_negative_labels)

pos_img_files, neg_img_files = read_filenames()

print( "Total Positive Images : " + str(len(pos_img_files)))
print("Total Negative Images : " + str(len(neg_img_files)))
print("Reading Images")

X, Y, pos_count, neg_count = read_images(pos_img_files, neg_img_files)

X = np.array(X)
Y = np.array(Y)

# X, Y = shuffle(X, Y, random_state=0)
pca = PCA(n_components=PCA_N_COMPONENTS) 
pca.fit(X)
X_pca = pca.transform(X)
joblib.dump(pca, 'pca106.pkl')

print("Images Read and Shuffled")
print("Positives: " + str(pos_count))
print("Negatives: " + str(neg_count))
print("Training Started")

clf1 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)

clf1.fit(X_pca, Y)
print("Trained")

joblib.dump(clf1, 'person_pre-eliminary06.pkl')

print("Hard Negative Mining")

winStride = (8, 8)
winSize = (64, 128)

print("Maximum Hard Negatives to Mine: " + str(MAX_HARD_NEGATIVES))

hard_negatives, hard_negative_labels = hard_negative_mine(neg_img_files, winSize, winStride)

pca2 = PCA(n_components=PCA_N_COMPONENTS) 
pca2.fit(hard_negatives)
hard_negatives_pca = pca2.transform(hard_negatives)
print("Final Samples Dims: " + str(hard_negatives.shape))
print("Retraining the classifier with final data")

clf2 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)

clf2.fit(hard_negatives_pca, hard_negative_labels)

print("Trained and Dumping")

joblib.dump(clf2, 'person_final06.pkl')
joblib.dump(pca2, 'pca206.pkl')