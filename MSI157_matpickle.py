import numpy as np
# import torch.utils.data as data
from sklearn.decomposition import PCA
import random
# import os
import pickle
import h5py
import hdf5storage
from sklearn import preprocessing
import scipy.io as sio


def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(groundTruth):
    labels_loc = {}
    m = 2
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        print('lenindices:',len(indices))

    whole_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]

    np.random.shuffle(whole_indices)
    return whole_indices


def load_data_HDF(image_file, label_file):
    image_data = hdf5storage.loadmat(image_file)
    label_data = hdf5storage.loadmat(label_file)
    data_all = image_data['train157']
    label = label_data['label']

    [nRow, nColumn, nBand] = data_all.shape
    print('barbara', nRow, nColumn, nBand)
    gt = label.reshape(np.prod(label.shape[:2]), )
    del image_data
    del label_data
    del label

    data_all = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    print(data_all.shape)
    data_scaler = preprocessing.scale(data_all)
    data_scaler = data_scaler.reshape(1024,1024,3)

    return data_scaler, gt

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    data_all = image_data['HypeRvieW']
    label = label_data['HypeRvieW']
    gt = label.reshape(np.prod(label.shape[:2]), )

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    print(data.shape)
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return data_scaler, gt

def getDataAndLabels(trainfn1, trainfn2, trainfn3):

    Data_Band_Scaler1, gt = load_data_HDF(trainfn1, trainfn3)
    Data_Band_Scaler2, gt = load_data_HDF(trainfn2, trainfn3)
    Data_Band_Scaler = Data_Band_Scaler2-Data_Band_Scaler1

    del trainfn1, trainfn2, trainfn3
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape


    patch_length = 4
    whole_data = Data_Band_Scaler
    padded_data = zeroPadding_3D(whole_data, patch_length)
    del Data_Band_Scaler

    np.random.seed(1334)

    whole_indices = sampling(gt)
    print('the whole indices', len(whole_indices))

    nSample = len(whole_indices)
    x = np.zeros((nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand))
    y = gt[whole_indices] ##label 0 1

    whole_assign = indexToAssignment(whole_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    print('indexToAssignment is ok')
    for i in range(len(whole_assign)):
        x[i] = selectNeighboringPatch(padded_data, whole_assign[i][0], whole_assign[i][1],patch_length)
    print('selectNeighboringPatch is ok')

    print(x.shape)
    del whole_assign
    del whole_data
    del padded_data

    imdb = {}
    imdb['data'] = np.zeros([nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand], dtype=np.float32)
    imdb['Labels'] = np.zeros([nSample], dtype=np.int64)
    imdb['set'] = np.zeros([nSample], dtype=np.int64)

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :, ] = x[iSample, :, :, :]
        imdb['Labels'][iSample] = y[iSample]


    imdb['set'] = np.ones([nSample]).astype(np.int64)
    print('Data is OK.')

    return imdb

train_data_file1 = 'D:\\Lsydata\\157\\train157_be.mat'
train_data_file2 = 'D:\\Lsydata\\157\\train157_af.mat'
train_label_file = 'D:\\Lsydata\\157\\gt.mat'

imdb = getDataAndLabels(train_data_file1, train_data_file2,train_label_file)

with open('datasets/MSI157_9_imdbgai_3.pickle', 'wb') as handle:
    pickle.dump(imdb, handle, protocol=4)

print('Images preprocessed')
