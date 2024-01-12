import torch
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
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

class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""


    def __init__(self, root, annotation_level: str = "variant", transform=None, target_transform=None, download=False):
        super(StanfordCars, self).__init__(root, transform=transform, target_transform=target_transform)
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        self.split_train = verify_str_arg("train", "split", ("train", "test"))
        self.split_test = verify_str_arg("test", "split", ("train", "test"))
        self._base_folder = os.path.join(root, "stanford_cars")
        self.devkit = os.path.join(self._base_folder, "devkit")

        self._annotations_mat_path_train = os.path.join(self.devkit, "cars_train_annos.mat")
        self._images_base_path_train = os.path.join(self._base_folder, "cars_train")
        self._annotations_mat_path_test = os.path.join(self._base_folder, "cars_test_annos_withlabels.mat")
        self._images_base_path_test = os.path.join(self._base_folder, "cars_test")

        self.nsessions = 10
        self.session = 0 

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples_train = [
            (
                os.path.join(self._images_base_path_train, annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path_train, squeeze_me=True)["annotations"]
        ]

        self._samples_test = [
            (
                os.path.join(self._images_base_path_test, annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path_test, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(os.path.join(self.devkit, "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        transforms_train = transforms.Compose([
                           transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                           ])
        self.data_train = []
        self.targets_train = []
        for image_path, target in self._samples_train:
            self.data_train.append(Image.open(image_path).convert("RGB"))
            self.targets_train.append(target)
            img = transforms_train(self.data_train[-1])

        self.targets_train = np.asarray(self.targets_train).astype(np.int64)

        self.data_test = []
        self.targets_test = []
        for image_path, target in self._samples_test:
            self.data_test.append(Image.open(image_path).convert("RGB"))
            self.targets_test.append(target)

        self.targets_test = np.asarray(self.targets_test).astype(np.int64)
    
    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.nsessions:
            raise StopIteration

        # loading train data
        if self.session == 0:
            index_train = np.arange(16)
        else:
            index_train = np.arange(16 + (self.session-1) * 20, 16 + self.session * 20)
            
        index_test = np.arange(16 + self.session * 20)

        train_x, train_y = self.SelectfromDefault(self.data_train, self.targets_train, index_train)

        # loading test data
        test_x, test_y = self.SelectfromDefault(self.data_test, self.targets_test, index_test)

        # Update state for next iter
        self.session += 1

        return (train_x, train_y, test_x, test_y)


    def SelectfromDefault(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if len(targets_tmp) == 0:
                targets_tmp = targets[ind_cl]
            else:
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

            for ind in ind_cl:
                data_tmp.append(data[ind])

        return data_tmp, targets_tmp


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not os.path.isdir(self.devkit):
            return False

        return os.path.exists(self._annotations_mat_path_train) and os.path.isdir(self._images_base_path_train) \
               and os.path.exists(self._annotations_mat_path_test) and os.path.isdir(self._images_base_path_test)



if __name__ == "__main__":
    
    dataroot = "/home/ap2313/rds/hpc-work/datasets"
    dataset = StanfordCars(root=dataroot)

    # loop over the training incremental batches
    count = 0
    for sess, session_data in enumerate(dataset):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y, test_x, test_y = session_data

        print("\n\n----------- Session {0} -------------".format(sess+1))
        print("train_x shape: {}, train_y shape: {}"
              .format(len(train_x), train_y.shape))
        print("Unique train_y: ", np.unique(train_y))

        print("\ntest_x shape: {}, train_y shape: {}"
              .format(len(test_x), test_y.shape))
        print("Unique test_y: ", np.unique(test_y))

        count += len(train_y)
    print("Train images:", count)

        # use the data
        #pass
    
    
   
