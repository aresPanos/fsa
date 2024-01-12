import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import sys

import torchvision.transforms as transforms
#import albumentations as albu

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

IMG_SIZE = 224

MEAN_DATASETS = {
            'MNIST':(0.1307,),
            'KMNIST':(0.1307,),
            'EMNIST':(0.1307,),
            'FashionMNIST':(0.1307,),
            'SVHN':  (0.4377,  0.4438,  0.4728),
            'CIFAR10':(0.4914, 0.4822, 0.4465),
            'CIFAR100':(0.5071, 0.4867, 0.4408),
            'CINIC10':(0.47889522, 0.47227842, 0.43047404),
            'TinyImagenet':(0.4802, 0.4481, 0.3975),
            'ImageNet100':(0.485, 0.456, 0.406),
            'ImageNet':(0.485, 0.456, 0.406),
        }

STD_DATASETS = {
    'MNIST':(0.3081,),
    'KMNIST':(0.3081,),
    'EMNIST':(0.3081,),
    'FashionMNIST':(0.3081,),
    'SVHN': (0.1969,  0.1999,  0.1958),
    'CIFAR10':(0.2023, 0.1994, 0.2010),
    'CIFAR100':(0.2675, 0.2565, 0.2761),
    'CINIC10':(0.24205776, 0.23828046, 0.25874835),
    'TinyImagenet':(0.2302, 0.2265, 0.2262),
    'ImageNet100':(0.229, 0.224, 0.225),
    'ImageNet':(0.229, 0.224, 0.225),
}

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, transform=None, target_transform=None, download=False):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.root = root
        self.nsessions = 10
        self.session = 0 

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


        self.data_train, self.data_test = [], []
        self.targets_train, self.targets_test = [], []

        # now load the picked numpy arrays
        for file_name, checksum in self.train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data_train.append(entry['data'])
                if 'labels' in entry:
                    self.targets_train.extend(entry['labels'])
                else:
                    self.targets_train.extend(entry['fine_labels'])

        self.data_train = np.vstack(self.data_train).reshape(-1, 3, 32, 32)
        self.data_train = self.data_train.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets_train = np.asarray(self.targets_train)


        # now load the picked numpy arrays
        for file_name, checksum in self.test_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data_test.append(entry['data'])
                if 'labels' in entry:
                    self.targets_test.extend(entry['labels'])
                else:
                    self.targets_test.extend(entry['fine_labels'])

        self.data_test = np.vstack(self.data_test).reshape(-1, 3, 32, 32)
        self.data_test = self.data_test.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets_test = np.asarray(self.targets_test)

        self._load_meta()

    
    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.nsessions:
            raise StopIteration

        # loading train data
        index_train = np.arange(self.session * 10, (self.session + 1) * 10)
        train_x, train_y = self.SelectfromDefault(self.data_train, self.targets_train, index_train)

        # loading test data
        index_test = np.arange((self.session + 1) * 10)
        test_x, test_y = self.SelectfromDefault(self.data_test, self.targets_test, index_test)

        # Update state for next iter
        self.session += 1

        return (train_x, train_y, test_x, test_y)


    def SelectfromDefault(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        return data_tmp, targets_tmp


    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            #print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

if __name__ == "__main__":
    
    
    dataroot = "/home/ap2313/rds/hpc-work/datasets"
    dataset = CIFAR100(root=dataroot)

    # loop over the training incremental batches
    for sess, session_data in enumerate(dataset):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y, test_x, test_y = session_data

        print("\n\n----------- Session {0} -------------".format(sess))
        print("train_x shape: {}, train_y shape: {}"
              .format(train_x.shape, train_y.shape))
        print("Unique train_y: ", np.unique(train_y))

        print("\ntest_x shape: {}, train_y shape: {}"
              .format(test_x.shape, test_y.shape))
        print("Unique test_y: ", np.unique(test_y))

        # use the data
        pass
    
   
