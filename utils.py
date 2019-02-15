import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image
import time


def load_data(dataset):
    """
    Parameters
    ----------
    dataset: the name of a dataset

    Returns
    ----------
    images_train: a tensor of C * N * C * W * H
    labels_train: a tensor of C * N * 1
    images_val: a tensor of C * N * C * W * H
    labels_val: a tensor of C * N * 1
    images_test: a tensor of C * N * C * W * H
    labels_test: a tensor of C * N * 1
    """

    if dataset == 'cifar100':
        num_class = 100
        num_sample_train = 500
        num_sample_test = 100
        num_sample_val = 0
        W = 32
        H = 32
        C = 3

        # load data
        with open('../data/cifar-100-python/meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            fine_label_names = data[b'fine_label_names']
            coarse_label_names = data[b'coarse_label_names']

        with open('../data/cifar-100-python/train', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            fine_labels_train = data[b'fine_labels']
            fine_labels_train = np.asarray(fine_labels_train)
            coarse_labels_train = data[b'coarse_labels']
            coarse_labels_train = np.asarray(coarse_labels_train)
            data_train = data[b'data']
            data_train = data_train.reshape((-1, 3, 32, 32))
            # data_train = np.asarray(data_train, dtype=np.float32)
            # data_train = data_train.transpose((0, 2, 3, 1))

        # plt.imshow(data_train[3])
        # plt.show()

        with open('../data/cifar-100-python/test', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            fine_labels_test = data[b'fine_labels']
            fine_labels_test = np.asarray(fine_labels_test)
            coarse_labels_test = data[b'coarse_labels']
            coarse_labels_test = np.asarray(coarse_labels_test)
            data_test = data[b'data']
            data_test = data_test.reshape((-1, 3, 32, 32))
            # data_test = data_test.transpose((0, 2, 3, 1))


        images_train = np.zeros((num_class, num_sample_train, C, W, H), dtype=np.uint8)
        labels_train = np.zeros((num_class, num_sample_train), dtype=int)
        images_val = 0
        labels_val = 0
        images_test = np.zeros((num_class, num_sample_test, C, W, H), dtype=np.uint8)
        labels_test = np.zeros((num_class, num_sample_test), dtype=int)

        for i in range(num_class):
            idx = fine_labels_train == i
            images_train[i] = data_train[idx]
            labels_train[i] = fine_labels_train[idx]

            idx = fine_labels_test == i
            images_test[i] = data_test[idx]
            labels_test[i] = fine_labels_test[idx]

        return images_train, labels_train, images_val, labels_val, images_test, labels_test

    elif dataset == 'imagenet':

        print('to do')
    else:
        print('Error: The dataset name is unknown.')



def build_network(network):
    """
    Parameters
    ----------
    network: the name of a network

    Returns
    ----------
    net: a network
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 96, 3)
            self.conv2 = nn.Conv2d(96, 128, 3)
            self.fc1 = nn.Linear(4608, 500)
            self.fc2 = nn.Linear(500, 200)
            self.fc3 = nn.Linear(200, 100)

        def dim_flat_features(self, x):
            size = x.shape[1:]
            dim = 1
            for s in size:
                dim *= s
            return dim

        def forward(self, x):
            # print('inside: sample size: %d' % x.shape[0])
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.dim_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    if network == 'resnet50':
        return 0

    elif network == 'simple':
        return Net()

    else:
        print('Error: The network name is unknown.')


def classifier(x, M, lab_map):
    """ iCaRL classifier
    Parameters
    ----------
    x: the testing features, n * d
    M: the mean vectors of exemplars, m * d

    Returns
    ----------
    lab: the labels of x, corresponding to M
    """

    n = np.shape(x)[0]
    m = np.shape(M)[0]
    x = np.tile(x[:, np.newaxis, :], (1, m, 1))
    M = np.tile(M[np.newaxis, :, :], (n, 1, 1))
    dis_2 = np.sum(np.square(x - M), axis=-1, keepdims=False)
    lab = np.argmin(dis_2, axis=1)
    lab = lab_map[lab]
    return lab

def norm_l2(x):
    """
    Parameters
    ----------
    x: the features, n * dimension

    Returns
    ----------
    x: the normalized features
    """
    n, d = np.shape(x)[0], np.shape(x)[1]
    norm = np.tile(np.sqrt(np.sum(np.square(x), keepdims=True)), (1, d))
    x = np.divide(x, norm)
    return x

def exemplar_selection(x, m):
    """
        Parameters
        ----------
        x: the features, n * dimension
        m: the number of selected exemplars

        Returns
        ----------
        pos_s: the position of selected exemplars
    """

    pos_s = []
    comb = 0
    mu = np.mean(x, axis=0, keepdims=False)
    for k in range(m):
        det = mu*(k+1) - comb
        dist = np.zeros(shape=(np.shape(x)[0]))
        for i in range(np.shape(x)[0]):
            if i in pos_s:
                dist[i] = np.inf
            else:
                dist[i] = np.linalg.norm(det - x[i])
        pos = np.argmin(dist)
        pos_s.append(pos)
        comb += x[pos]

    return pos_s


def one_hot(y, c):
    """
        Parameters
        ----------
        y: the original labels, size = (n)
        c: the total number of classes

        Returns
        ----------
        y_onehot: the one-hot labels
    """
    y = np.array(y, dtype=np.int)
    n = np.shape(y)[0]
    y_onehot = np.zeros(shape=(n, c),dtype=np.int)
    for i in range(n):
        y_onehot[i, y[i]] = 1
    return y_onehot


def data_augmentation_e2e(img, lab):
    """
        Realize the data augmentation in End-to-End paper
        Parameters
        ----------
        img: the original images, size = (n, c, w, h)
        lab: the original labels, size = (n)

        Returns
        ----------
        img_aug: the original images, size = (n * 12, c, w, h)
        lab_aug: the original labels, size = (n * 12)
    """
    shape = np.shape(img)
    img_aug = np.zeros((shape[0], 12, shape[1], shape[2], shape[3]))
    img_aug[:, 0, :, :, :] = img
    lab_aug = np.zeros((shape[0], 12))

    for i in range(shape[0]):
        np.random.seed(int(time.time()) % 1000)

        im = img[i]

        # brightness
        brightness = (np.random.rand(1)-0.5)*2*63
        im_temp = im + brightness

        img_aug[i, 1] = im_temp

        # constrast
        constrast = (np.random.rand(1)-0.5)*2*0.8+1
        m0 = np.mean(im[0])
        m1 = np.mean(im[1])
        m2 = np.mean(im[2])
        im_temp = im
        im_temp[0] = (im_temp[0]-m0)*constrast + m0
        im_temp[1] = (im_temp[1]-m1)*constrast + m1
        im_temp[2] = (im_temp[2]-m2)*constrast + m2
        img_aug[i, 2] = im_temp

        # crop
        im_temp = img_aug[i, :3]
        for j in range(3):
            x_ = int(np.random.rand(1)*1000)%8
            y_ = int(np.random.rand(1)*1000)%8
            im_temp = np.zeros(shape=(shape[1], shape[2]+8, shape[3]+8))
            im_temp[:, 4:-4, 4:-4] = img_aug[i, j]
            img_aug[i, 3+j] = im_temp[:, x_:x_+shape[2], y_:y_+shape[3]]



        # mirror
        for j in range(6):
            im_temp = img_aug[i, j]
            img_aug[i, 6 + j] = im_temp[:,-1::-1,:]

        lab_aug[i, :] = lab[i]

    idx = np.where(img_aug>255)
    img_aug[idx] = 255
    idx = np.where(img_aug<0)
    img_aug[idx] = 0

    img_aug = np.reshape(img_aug, newshape=(shape[0]*12, shape[1], shape[2], shape[3]))
    img_aug = np.array(img_aug, dtype=np.uint8)
    lab_aug = np.reshape(lab_aug, newshape=(shape[0]*12))
    lab_aug = np.array(lab_aug, dtype=np.int32)
    return img_aug, lab_aug


class iCaRL_loss(nn.Module):
    def __init__(self):
        super(iCaRL_loss, self).__init__()
        # self.logist = logist
        # self.target = target

    def forward(self, logist, target):
        eps = 0.000001
        logist = logist.double()
        target = target.double()

        p0 = torch.mul(target, torch.log(logist + eps))
        p1 = torch.mul(1-target, torch.log(1-logist + eps))
        loss = - torch.add(p0, p1)
        loss = torch.sum(loss)
        return loss



im_mean = np.array([129.37731888, 124.10583864, 112.47758569], dtype= np.float)
im_std = np.array([68.20947949, 65.43124043, 70.45866994], dtype= np.float)
train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(), transforms.Normalize(im_mean, im_std)])

# train_transform = transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
#      transforms.RandomVerticalFlip(), transforms.RandomRotation(degrees=360),
#      transforms.ToTensor(), transforms.Normalize(im_mean, im_std)])

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(im_mean, im_std)])

# transform_flip_crop = transforms.Compose()

def img_transform(img, flag):
    np.random.seed(int(time.time()*1000%100))
    if flag == 'train':
        transform = train_transform
        # if np.random.rand(1)> 0.1:
        #     print('do')
        #     transform = train_transform
        # else:
        #     print('not do')
        #     transform = test_transform
    elif flag == 'test':
        transform = test_transform
    # elif flag == 'flip_crop':
    #     transform = transform_flip_crop
    else:
        print('img_transform parameter error')
        return img

    shape = np.shape(img)
    img_tf = np.zeros(shape=shape)
    img = np.transpose(img, axes=(0, 2, 3, 1))
    for i in range(shape[0]):
        im = Image.fromarray(img[i])
        img_tf[i] = transform(im)
    return img_tf
