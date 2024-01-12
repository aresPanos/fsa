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

class FGVCAircraft(VisionDataset):
    """`FGVC Aircraft <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    The dataset contains 10,000 images of aircraft, with 100 images for each of 100
    different aircraft model variants, most of which are airplanes.
    Aircraft models are organized in a three-levels hierarchy. The three levels, from
    finer to coarser, are:

    - ``variant``, e.g. Boeing 737-700. A variant collapses all the models that are visually
        indistinguishable into one class. The dataset comprises 100 different variants.
    - ``family``, e.g. Boeing 737. The dataset comprises 70 different families.
    - ``manufacturer``, e.g. Boeing. The dataset comprises 30 different manufacturers.

    Args:
        root (string): Root directory of the FGVC Aircraft dataset.
        split (string, optional): The dataset split, supports ``train``, ``val``,
            ``trainval`` and ``test``.
        annotation_level (str, optional): The annotation level, supports ``variant``,
            ``family`` and ``manufacturer``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

    def __init__(self, root, annotation_level: str = "variant", transform=None, target_transform=None, download=False):
        root = os.path.join(root, "fgvc_aircraft")
        super(FGVCAircraft, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split_train = verify_str_arg("trainval", "split", ("train", "val", "trainval", "test"))
        self.split_test = verify_str_arg("test", "split", ("train", "val", "trainval", "test"))

        self.nsessions = 10
        self.session = 0 

        self._annotation_level = verify_str_arg(
            annotation_level, "annotation_level", ("variant", "family", "manufacturer")
        )

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = os.path.join(
            self._data_path,
            "data",
            {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }[self._annotation_level],
        )
        with open(annotation_file, "r") as f:
            self.classes = [line.strip() for line in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        image_data_folder = os.path.join(self._data_path, "data", "images")
        labels_file_train = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self.split_train}.txt")
        labels_file_test = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self.split_test}.txt")

        self.data_train = []
        self.targets_train = []
        with open(labels_file_train, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self.data_train.append(Image.open(os.path.join(image_data_folder, f"{image_name}.jpg")).convert("RGB"))
                self.targets_train.append(self.class_to_idx[label_name])

        self.targets_train = np.asarray(self.targets_train).astype(np.int64)


        self.data_test = []
        self.targets_test = []
        with open(labels_file_test, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self.data_test.append(Image.open(os.path.join(image_data_folder, f"{image_name}.jpg")).convert("RGB"))
                self.targets_test.append(self.class_to_idx[label_name])

        self.targets_test = np.asarray(self.targets_test).astype(np.int64)
    
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
            if len(targets_tmp) == 0:
                targets_tmp = targets[ind_cl]
            else:
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

            for ind in ind_cl:
                data_tmp.append(data[ind])

        return data_tmp, targets_tmp


    def _download(self) -> None:
        """
        Download the FGVC Aircraft dataset archive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, self.root)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)



if __name__ == "__main__":
    
    dataroot = "/home/ap2313/rds/hpc-work/datasets"
    dataset = FGVCAircraft(root=dataroot)

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
    
    
   
