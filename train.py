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

MAX_HARD_NEGATIVES = 200
PCA_N_COMPONENTS = 0.95
parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--pos', help='Path to directory containing Positive Images')
parser.add_argument('--neg', help='Path to directory containing Negative images')

args = parser.parse_args()
pos_img_dir = args.pos
neg_img_dir = args.neg


def crop_centre(img):
    h, w = img.shape[0], img.shape[1]
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
            features_pca = pca.transform(features.reshape(1, -1))  # Apply PCA
            if (clf1.predict(features_pca) == 1):
                hard_negatives.append(features_pca)
                hard_negative_labels.append(0)
                count = count + 1
            if (count == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)
        num = num + 1
        sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %")
        sys.stdout.flush()
    return np.array(hard_negatives), np.array(hard_negative_labels)

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
        features_pca = pca.transform(features.reshape(1, -1))  # Apply PCA
        pos_count += 1
        X.append(features_pca)
        Y.append(1)
    neg_count = 0
    for img_file in neg_files:
        print(os.path.join(neg_img_dir, img_file))
        img = cv2.imread(os.path.join(neg_img_dir, img_file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        windows = ten_random_windows(gray_img)
        for win in windows:
            features = hog(win, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            features_pca = pca.transform(features.reshape(1, -1))  # Apply PCA
            neg_count += 1
            X.append(features_pca)
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
            if hard_negatives:
                hard_negatives_transformed = pca.transform(hard_negatives)
                if (clf1.predict([hard_negatives_transformed]) == 1):
                    hard_negatives.append(features)
                    hard_negative_labels.append(0)
                    count = count + 1
            if (count == MAX_HARD_NEGATIVES):
                return np.array(hard_negatives), np.array(hard_negative_labels)

        num = num + 1

        sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(round((count / float(MAX_HARD_NEGATIVES))*100, 4)) + " %")

        sys.stdout.flush()

    return np.array(hard_negatives), np.array(hard_negative_labels)

def extract_features_for_pca(pos_files, neg_files):
    features = []
    for img_file in pos_files + neg_files:
        img_path = os.path.join(pos_img_dir if img_file in pos_files else neg_img_dir, img_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # You can modify this part based on how you want to handle positive and negative images
        cropped = crop_centre(gray) if img_file in pos_files else ten_random_windows(gray)
        for crop in cropped:
            if crop.shape[0] < 128 or crop.shape[1] < 64:  # Replace with actual minimum dimensions required
                continue  # Skip this crop if it's too small
            hog_features = hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2", transform_sqrt=True, feature_vector=True)
            features.append(hog_features)
    return np.array(features)


pos_img_files, neg_img_files = read_filenames()

print( "Total Positive Images : " + str(len(pos_img_files)))
print("Total Negative Images : " + str(len(neg_img_files)))
print("Reading Images")
pca = PCA(n_components=PCA_N_COMPONENTS) 
features_for_pca = extract_features_for_pca(pos_img_files, neg_img_files)
pca.fit(features_for_pca)

X, Y, pos_count, neg_count = read_images(pos_img_files, neg_img_files)

X = np.array(X)
Y = np.array(Y)

X, Y = shuffle(X, Y, random_state=0)

print("Images Read and Shuffled")
print("Positives: " + str(pos_count))
print("Negatives: " + str(neg_count))
print("Training Started")

clf1 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)
X_flattened = [x.flatten() for x in X]  # Flatten each feature vector
X_array = np.array(X_flattened) 
clf1.fit(X_array, Y)
print("Trained")

joblib.dump(clf1, 'person_pre_eliminary_200/20000.pkl')

print("Hard Negative Mining")

winStride = (8, 8)
winSize = (64, 128)

print("Maximum Hard Negatives to Mine: " + str(MAX_HARD_NEGATIVES))

hard_negatives, hard_negative_labels = hard_negative_mine(neg_img_files, winSize, winStride)

sys.stdout.write("\n")
hard_negatives = np.array([hn.flatten() for hn in hard_negatives])
hard_negatives = np.concatenate((hard_negatives, X_array), axis=0)
hard_negative_labels = np.concatenate((hard_negative_labels, Y), axis=0)

hard_negatives, hard_negative_labels = shuffle(hard_negatives, hard_negative_labels, random_state=0)

print("Final Samples Dims: " + str(hard_negatives.shape))
print("Retraining the classifier with final data")
hard_negatives_pca = pca.transform(hard_negatives)
clf2 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)

# clf2.fit(hard_negatives, hard_negative_labels)
clf2.fit(hard_negatives_pca, hard_negative_labels)

print("Trained and Dumping")

joblib.dump(clf2, 'person_final_200/20000.pkl')
