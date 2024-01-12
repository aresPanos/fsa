import pickle
import os
import os.path
from torchvision.io import read_image
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

class Letters(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None):
        root = os.path.join(root, "letters") # /English
        super(Letters, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.nsessions = 11
        self.session = 0 
        
        #self.data_train, self.targets_train, self.data_test, self.targets_test = self._get_train_test_data()
        with open(root + '/full_dataset.pickle', 'rb') as handle:
            all_data = pickle.load(handle)

        self.data_train, self.targets_train = all_data["data_train"], all_data["targets_train"]
        self.data_test, self.targets_test = all_data["data_test"], all_data["targets_test"]

    
    def _get_train_test_data(self):
        num_classes = 62
        dir_category = ["Img/GoodImg/Bmp", "Hnd/Img", "Fnt"]
        np.random.seed(1234)
        data_train, targets_train, data_test, targets_test = [], [], [], []
        for categ in dir_category:
            images_per_class = np.zeros(num_classes, dtype=np.int64)
            for cl in range(num_classes):
                cl_name = "Sample00" + str(cl+1) if cl < 9 else "Sample0" + str(cl+1)
                path_class = os.path.join(self.root, categ, cl_name)
                cls_num_imgs = len(os.listdir(path_class))
                train_cls_num_imgs = int(0.8*cls_num_imgs)
                images_per_class[cl] = cls_num_imgs

                random_perm = np.random.permutation(cls_num_imgs)
                train_split, test_split = random_perm[:train_cls_num_imgs], random_perm[train_cls_num_imgs:]

                for idx, f in enumerate(os.listdir(path_class)):
                    img_path = os.path.join(path_class, f)
                    if os.path.isfile(img_path):
                        if idx in train_split or idx in test_split:
                            img = read_image(img_path)

                            if idx in train_split:
                                data_train.append(img)
                                targets_train.append(cl)
                            elif idx in test_split:
                                data_test.append(img)
                                targets_test.append(cl)

        targets_train = np.asarray(targets_train).astype(np.int64)
        targets_test = np.asarray(targets_test).astype(np.int64)

        return data_train, targets_train, data_test, targets_test      


    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.nsessions:
            raise StopIteration

        # loading train data
        index_train = np.arange(12) if self.session == 0 else 12 + np.arange((self.session-1) * 5, self.session * 5)
        train_x, train_y = self.SelectfromDefault(self.data_train, self.targets_train, index_train)

        # loading test data
        index_test = np.arange(12) if self.session == 0 else np.arange(12 + self.session * 5)
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




if __name__ == "__main__":
    
    dataroot = "/home/ap2313/rds/hpc-work/datasets"
    dataset = Letters(root=dataroot)

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
    
    
   
