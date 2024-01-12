import torch
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pickle
import sys

import torchvision.transforms as transforms
#import albumentations as albu

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg

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

class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(self, root, transform=None, target_transform=None, download=False):
        root = os.path.join(root, "svhn")
        super(SVHN, self).__init__(root, transform=transform, target_transform=target_transform)
        split_train = "train"
        self.split_train = verify_str_arg(split_train, "split", tuple(self.split_list.keys()))
        self.url_train = self.split_list[split_train][0]
        self.filename_train = self.split_list[split_train][1]
        self.file_md5_train = self.split_list[split_train][2]

        split_test = "test"
        self.split_test = verify_str_arg(split_test, "split", tuple(self.split_list.keys()))
        self.url_test = self.split_list[split_test][0]
        self.filename_test = self.split_list[split_test][1]
        self.file_md5_test = self.split_list[split_test][2]

        self.nsessions = 5
        self.session = 0 

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat_train = sio.loadmat(os.path.join(self.root, self.filename_train))
        loaded_mat_test = sio.loadmat(os.path.join(self.root, self.filename_test))

        self.data_train = loaded_mat_train["X"]
        self.data_test = loaded_mat_test["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets_train = loaded_mat_train["y"].astype(np.int64).squeeze()
        self.targets_test = loaded_mat_test["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets_train, self.targets_train == 10, 0)
        np.place(self.targets_test, self.targets_test == 10, 0)
        self.data_train = np.transpose(self.data_train, (3, 2, 0, 1))
        self.data_test = np.transpose(self.data_test, (3, 2, 0, 1))

        self.data_train = np.transpose(self.data_train, (0, 2, 3, 1))
        self.data_test = np.transpose(self.data_test, (0, 2, 3, 1))

    
    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.nsessions:
            raise StopIteration

        # loading train data
        index_train = np.arange(self.session * 2, (self.session + 1) * 2)
        train_x, train_y = self.SelectfromDefault(self.data_train, self.targets_train, index_train)

        # loading test data
        index_test = np.arange((self.session + 1) * 2)
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


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5_train = self.split_list[self.split_train][2]
        fpath_train = os.path.join(root, self.filename_train)
        md5_test = self.split_list[self.split_test][2]
        fpath_test = os.path.join(root, self.filename_test)

        return check_integrity(fpath_train, md5_train) and check_integrity(fpath_test, md5_test)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)



if __name__ == "__main__":
    
    dataroot = "/home/ap2313/rds/hpc-work/datasets"
    dataset = SVHN(root=dataroot)

    # loop over the training incremental batches
    count = 0
    for sess, session_data in enumerate(dataset):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y, test_x, test_y = session_data

        print("\n\n----------- Session {0} -------------".format(sess+1))
        print("train_x shape: {}, train_y shape: {}"
              .format(train_x.shape, train_y.shape))
        print("Unique train_y: ", np.unique(train_y))

        print("\ntest_x shape: {}, train_y shape: {}"
              .format(test_x.shape, test_y.shape))
        print("Unique test_y: ", np.unique(test_y))

        count += len(train_y)
    print("Train images:", count)

        # use the data
        #pass
    
    
   
