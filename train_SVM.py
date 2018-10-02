# Import as usual
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from scipy.ndimage.measurements import label
from sklearn.utils import shuffle
from skimage.feature import hog
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time
import pickle
import cv2
import glob
import random

load_small = False # for prototyping

def load_img(img_file):
    if img_file[-4:] in ('jpeg', 'JPEG', '.jpg', '.JPG'):
        img = (mpimg.imread(img_file) / 255.).astype(np.float32)
    elif img_file[-4:] in ('.png', '.PNG'):
        img = mpimg.imread(img_file)
    else:
        raise ValueError('Unrecognized image format: {}'.format(img_file[-4:]))
    return img

if load_small:
    vehicle_files = glob.glob('data/vehicles_smallset/*')
    non_veh_files = glob.glob('data/non-vehicles_smallset/*')
else:
    vehicle_files = glob.glob('data/vehicles/KITTI_extracted/*')
    vehicle_files.extend(glob.glob('data/vehicles/GTI_Far/*'))
    vehicle_files.extend(glob.glob('data/vehicles/GTI_Left/*'))
    vehicle_files.extend(glob.glob('data/vehicles/GTI_MiddleClose/*'))
    vehicle_files.extend(glob.glob('data/vehicles/GTI_Right/*'))
    
    non_veh_files = glob.glob('data/non-vehicles/Extras/*')
    non_veh_files.extend(glob.glob('data/non-vehicles/GTI/*'))

n_veh = len(vehicle_files)
n_non = len(non_veh_files)
print('number of pictures: {}'.format(n_veh+n_non))

# raw downsampled pixel features
def bin_spatial(img, output_space='RGB', size=32):
    
    # color conversion dictionary
    color_dict = {'HLS': cv2.COLOR_RGB2HLS,
        'HSV': cv2.COLOR_RGB2HSV,
        'BGR': cv2.COLOR_RGB2BGR,
        'LUV': cv2.COLOR_RGB2LUV,
        }
    
    # convert RGB to output space
    if output_space != 'RGB': img = cv2.cvtColor(img, color_dict[output_space])
    # downsample
    features = cv2.resize(img, (size, size)).ravel()
    
    return features

# color histogram features
def color_hist(img, output_space='RGB', nbins=32, bins_range=(0, 256)):
    
    # color conversion dictionary
    color_dict = {'HLS': cv2.COLOR_RGB2HLS,
        'HSV': cv2.COLOR_RGB2HSV,
        'BGR': cv2.COLOR_RGB2BGR,
        'LUV': cv2.COLOR_RGB2LUV,
        }
    
    # convert RGB to output space
    if output_space != 'RGB': img = cv2.cvtColor(img, color_dict[output_space])
    
    # Compute the histogram of the RGB channels separately
    hist1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    hist2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    hist3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    # Concatenate the histograms into a single feature vector
    features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    
    return features

# HOG features
def get_hog_features(img, output_space, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, 
                     feature_vec=True):
    
    if output_space == 'gray': img_1c = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: raise ValueError('hog feature detracter only implemented for gray scale!')
    
    return  hog(img_1c, orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block),
                block_norm='L2-Hys', visualise=vis,
                feature_vector=feature_vec)


# feature extracter from image file names, combining above feature methods
def extract_features(img_files, spatial, hist, hog, verbose=False):
    features = []
    color_dict = {"HLS": cv2.COLOR_RGB2HLS,
        "HSV": cv2.COLOR_RGB2HSV,
        "BGR": cv2.COLOR_RGB2BGR,
        "LUV": cv2.COLOR_RGB2LUV,
        }
    for img_file in (img_files):
        
        # load in image in correct format
        img = load_img(img_file)
        
        # spatial color features
        if spatial['use']: spa_features  = bin_spatial(img, output_space=spatial['cspace'], size=spatial['size'])
        
        # colorchannel histogram features
        if hist['use']: hist_features = color_hist(img, output_space=hist['cspace'],
                                                   nbins=hist['bins'], bins_range=hist['range'])
        # HOG features
        if hog['use']: hog_features  = get_hog_features(img, output_space=hog['cspace'],
                                                        orient=hog['orient'], pix_per_cell=hog['pix_per_cell'],
                                                        cell_per_block=hog['cell_per_block'],
                                                        vis=False, feature_vec=True)
        combined_feat = np.array([])
        if spatial['use']: combined_feat = np.concatenate((combined_feat, spa_features ))
        if hist['use']:    combined_feat = np.concatenate((combined_feat, hist_features))
        if hog['use']:     combined_feat = np.concatenate((combined_feat, hog_features ))
        # Append the new feature vector to the features list
        features.append(combined_feat)

    if verbose: print('features returned with:')
    if spatial['use']: print('{} spatial color features'.format(len(spa_features)))
    if hist['use']:    print('{} color channel histogram features'.format(len(hist_features)))
    if hog['use']:     print('{} HOG features'.format(len(hog_features)))
    
    return features

# spatial binning hyperparams
spatial_params = {
    'use':    False,
    'cspace': 'RGB',
    'size':   32}

# color channel histogram hyperparams
hist_params = {
    'use':    True,
    'cspace': 'RGB',
    'bins':   16,
    'range':  (0, 256)}

# HOG hyperparams
hog_params = {
    'use':            True,
    'cspace':         'gray',
    'orient':         9,
    'pix_per_cell':   8,
    'cell_per_block': 2}

# combine hyperparams in one dictionary
feat_params_dict = {
    'spatial': spatial_params,
    'hist':    hist_params,
    'hog':     hog_params}

# actually extract the features
car_features     = extract_features(vehicle_files, spatial_params, hist_params, hog_params, verbose=True)
non_car_features = extract_features(non_veh_files, spatial_params, hist_params, hog_params)

X = np.vstack((car_features, non_car_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

# hyperparameters
val_size = 0.2 # fraction of data separated into valiation set

# just in case shuffle, although shouldn't be necessary because train_test_split() includes shuffling as well
X, y = shuffle(X, y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

# create normalizer based on *only* training data
feat_scaler = StandardScaler().fit(X_train)

# normalize training and validation set (and delete )
X_train = feat_scaler.transform(X_train);
X_val   = feat_scaler.transform(X_val);
print('Amount of training samples: {}, amount of validation samples: {}'.format(X_train.shape[0],
                                                                                X_val.shape[0]))

print('starts training')
hyper_params = {
    'kernel': ('rbf', 'poly'), # ('poly', 'linear', 'rbf', 'sigmoid'),
    'C':      [0.001, 0.1, 1, 10, 100], #[0.001, 0.05, 0.1, 0.5, 1, 10, 100]
    'gamma':  [0.001, 0.1, 1, 10, 100] #[0.001, 0.05, 0.1, 0.5, 1, 10, 100]
}
svr = svm.SVC(kernel='poly')
clf = GridSearchCV(svr, hyper_params)

# train for all hyper parameter combinations
tic = time.time()
clf.fit(X_train, y_train)
print(clf.estimator.kernel)
print('Optimal parameters for classifier are ', clf.best_params_)
print('Training for all hyperparameter combinations took {:.0f} minutes'.format((time.time()-tic)/60))

print('Validation accuracy: {:.3f}'.format(clf.score(X_val, y_val)))

model_data = {}
model_data['classifier'] = clf
model_data['preprocessing_feature_scaler'] = feat_scaler
model_data['feature_params'] = feat_params_dict
pickle.dump( model_data, open( "model_data/model_data_pickle.p", "wb" ) )



