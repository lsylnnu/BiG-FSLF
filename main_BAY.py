import os
import numpy as np
import random
import torch.nn.functional as F
import torch
import torch.utils.data as dataf
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import matplotlib.pyplot as plt
from scipy import io
import sys
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 3) #VHR==3
parser.add_argument("-d","--tar_input_dim",type = int, default = 224) #bay==224
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-sum","--sum_num_per_class",type = int, default = 20)
parser.add_argument("-e","--episode",type = int, default= 1000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m" ,"--test_class_num",type=int, default=2)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
SUM_NUM_PER_CLASS = args.sum_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class ##2
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class

patchsize = 9

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets',  'MSI157_9_imdbgai_3.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']
labels_train = source_imdb['Labels']
keys_all_train = sorted(list(set(labels_train)))  # class [0,1]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
data = utils.sanity_check500(data) # 500 labels samples per class
print("Num classes of the number of class larger than 500 in dataset: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose


metatrain_data= data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data


print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'])

source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=SUM_NUM_PER_CLASS*CLASS_NUM, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
test_data1 = 'datasets/bayarea/BayArea_before.mat'
test_data2 = 'datasets/bayarea/BayArea_after.mat'
test_label = 'datasets/bayarea/bayArea_gtChanges2.mat'

Data_Band_Scaler1,GroundTruth = utils.load_data(test_data1, test_label)
Data_Band_Scaler2,GroundTruth = utils.load_data(test_data2, test_label)
Data_Band_Scaler = Data_Band_Scaler2-Data_Band_Scaler1



# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) ## (600，500，224)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth)) ##2
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn+ HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 2
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS  ##5
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class ##5
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices1:', len(train_indices))  # 5
    print('the number of test_indices1:', len(test_indices))  # 9693
    print('the number of train_indices1 after data argumentation:', len(da_train_indices))  # 200
    print('labeled sample indices1:',train_indices)

    nTrain = len(train_indices) ##5
    nTest = len(test_indices) ##9693
    da_nTrain = len(da_train_indices) ##200

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')


    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
                                                       Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :]
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)
        #     utils.radiation_noise(
        #     data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
        #     Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        # imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('imdb_da_train ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain

def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 1 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=SUM_NUM_PER_CLASS*CLASS_NUM, shuffle=True, num_workers=0)
    del target_dataset


    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)
        # self.sig = nn.Sigmoid()
        self.sig = nn.ReLU()
        # self.soft = nn.Softmax

    def forward(self, input):
        x = self.conv1(input)
        # print('x.conv1:',x.shape)
        x = self.conv2(x)
        # print('x.conv2:',x.shape)
        x = self.conv3(x)
        # print('x.conv3:',x.shape)
        x = self.conv4(x)
        # print('x.conv4:',x.shape)
        x = self.conv5(x)
        # print('x.conv5:',x.shape)
        x = self.conv6(x)
        # print('x.conv6:', x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sig(x)
        # x = nn.Softmax(dim=1)(x)
        return x



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Upsample(scale_factor=3),
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.dconv2 = nn.Sequential(
            # nn.Upsample(scale_factor=3)
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            # nn.Sigmoid(),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.dconv1(x)
        x = self.dconv2(x)

        return x


def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1+x3, inplace=True)
        return out


class D_Res_3d_CNN(nn.Module): ##1 8 16
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160
        # self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM, bias=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        # nn.BatchNorm3d(128),

        x = self.block2(x)
        x = self.maxpool2(x)

        x = self.conv(x)

        # y = self.classifier(x)
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        # print('x.preconv:',x.shape)
        x = self.preconv_bn(x)
        # print('x.preconv_bn:', x.shape)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.encoder = Encoder()
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION) ##3,100

    def forward(self,x, domain='source'):
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)

        feature = self.feature_encoder(x)
        output = self.encoder(feature)

        feature = feature.view(feature.shape[0], -1)
        output = output.view(output.size(0), -1)

        return feature, output



crossEntropy = nn.CrossEntropyLoss().cuda() ##FSL
loss_fun2 = nn.MSELoss() ##GAN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits



nDataSet =1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_kappa = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None


seeds = [ 1330,1226,1336,1220,  1337, 1334, 1236,  1235, 1228, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    # model
    feature_encoder = Network()
    dis = Discriminator()

    feature_encoder.apply(weights_init)
    dis.apply(weights_init)

    feature_encoder.cuda()
    dis.cuda()

    feature_encoder.train()
    dis.train()

    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    dis_optim = torch.optim.Adam(dis.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    last_kappa = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(800):
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()


        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)


            supports, support_labels = support_dataloader.__iter__().next()  ## supports(2, 3, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  ## querys(38,3,9,9)  支持集和查询集 1:19 两类*2

            # calculate features
            support_features, support_outputs = feature_encoder(supports.cuda())
            query_features, query_outputs = feature_encoder(querys.cuda())
            target_features, target_outputs = feature_encoder(target_data.cuda(),domain='target')

            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features

            '''Discriminator'''
            g_features = torch.cat([support_outputs, query_outputs], dim=0)

            fake = dis(g_features.view(g_features.size(0)*15, 128, 3, 3)).mean()
            real = dis(target_outputs.view(target_outputs.size(0)*15, 128, 3,3)).mean()

            d_loss = 1 - real + fake

            '''Generator'''
            a_loss = torch.mean(1 - fake)
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda().long())
            g_loss = f_loss+ 0.01 * a_loss

            # Update parameters
            dis.zero_grad()
            feature_encoder.zero_grad()
            d_loss.backward(retain_graph=True)
            g_loss.backward()
            dis_optim.step()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels.long()).item()
            total_num += querys.shape[0]



        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()
            querys, query_labels = query_dataloader.__iter__().next()

            # calculate features
            support_features, supports_outputs = feature_encoder(supports.cuda(), domain='target')
            query_features, query_outputs = feature_encoder(querys.cuda(), domain='target')
            source_features, source_outputs = feature_encoder(source_data.cuda())

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features

            '''Discrimer'''
            g_features = torch.cat([supports_outputs, query_outputs], dim=0)

            fake = dis(g_features.view(g_features.size(0) * 15, 128, 3, 3)).mean()
            real = dis(source_outputs.view(source_outputs.size(0) * 15, 128, 3, 3)).mean()


            d_loss = 1 - real + fake

            '''Generation'''
            a_loss = torch.mean(1 - fake)
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda().long())

            g_loss = f_loss+ 0.01 * a_loss


            # Update parameters
            dis.zero_grad()
            feature_encoder.zero_grad()
            d_loss.backward(retain_graph=True)
            g_loss.backward()
            dis_optim.step()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels.long()).item()
            total_num += querys.shape[0]

        if (episode + 1) % 10 == 0:  # display
            print('episode {:>3d}:  Dis loss: {:6.4f}, Gen loss: {:6.4f}, acc {:6.4f}'.format(episode + 1, d_loss.item(),g_loss.item(),total_hit / total_num, ))


        '''----TEST----'''
        if (episode + 1) % 50 == 0 or episode==0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            train_features , _ = feature_encoder(Variable(train_datas).cuda(), domain='target')

            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target')
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            test_kappa = metrics.cohen_kappa_score(labels, predict)

            print('\t\tkappa:', test_kappa)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_kappa > last_kappa:
                # save networks
                torch.save(feature_encoder.state_dict(),str("checkpoints/BiGFSLF_feature_encoder_" + "BAY_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                print("save networks for episode:",episode+1)
                last_kappa = test_kappa
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}, best kappa={}'.format(best_episdoe + 1, last_accuracy, last_kappa))

    if test_kappa > best_kappa:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}, best kappa={}'.format(iDataSet, best_episdoe + 1, last_accuracy,last_kappa))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))



#################classification map################################

for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

print('best_G.max:',int(np.max(best_G)))
print('best_G.min:',int(np.min(best_G)))

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0.5, 0.5, 0.5]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [1, 1, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/BAY_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
