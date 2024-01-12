import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pickle
import sys
from torchvision.datasets.vision import VisionDataset
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_datasets as tfds


def limit_tensorflow_memory_usage(gpu_memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]
                )
        except RuntimeError as e:
            print(e)

limit_tensorflow_memory_usage(1024)


class DSprites(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, seed: int=1):
        super(DSprites, self).__init__(root, transform=transform, target_transform=target_transform)
        self.nsessions = 7
        self.session = 0 
        self.train_size = 100_000
        self.test_size = 12_000

        tf.compat.v1.enable_eager_execution()
        train_split = "train[:%d]" %self.train_size
        decoders = None

        ds_train = tfds.load(
            "dsprites",
            split=train_split,
            shuffle_files=True,
            data_dir=root,
            with_info=False,
            decoders=decoders,
            read_config=tfds.ReadConfig(shuffle_seed=seed)
        )
        
        test_split = 'train[%d:%d]'%(self.train_size, self.train_size + self.test_size)
        ds_test = tfds.load(
            "dsprites",
            split=test_split,
            shuffle_files=True,
            data_dir=root,
            with_info=False,
            decoders=None,
            read_config=tfds.ReadConfig(shuffle_seed=seed)
        )

        train_iterator = ds_train.as_numpy_iterator()
        test_iterator = ds_test.as_numpy_iterator()

        self.data_train, self.targets_train = self._get_data_from_iterator(train_iterator, self.train_size)
        self.data_test, self.targets_test = self._get_data_from_iterator(test_iterator, self.test_size)
        

    def _get_data_from_iterator(self, iterator, num_examples):
        images, labels = [], []
        for _ in range(num_examples):
            try:
                item = iterator.next()
            except StopIteration:  # the last batch may be less than batch_size
                break

            images.append(255.0 * item['image'].squeeze()[None, :, :])
            labels.append(self._get_label(item))

        images = np.vstack(images).astype(np.float32)
        labels = np.asarray(labels).astype(np.int64)

        return images, labels


    def _get_label(self, item):
        num_classes = 16
        predicted_attribute = 'label_x_position'
        num_original_classes = 32
        class_division_factor = float(num_original_classes) / float(num_classes)

        return np.floor(float(item[predicted_attribute]) / class_division_factor)

        
    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.nsessions:
            raise StopIteration

        # loading train data
        index_train = np.arange(4) if self.session == 0 else 4 + np.arange((self.session-1) * 2, self.session * 2)
        train_x, train_y = self.SelectfromDefault(self.data_train, self.targets_train, index_train)

        # loading test data
        index_test = np.arange(4) if self.session == 0 else np.arange(4 + self.session * 2)
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
    dataset = DSprites(root=dataroot)
    trsf = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Grayscale(num_output_channels=3),
                          transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                          ])

    # loop over the training incremental batches
    count = 0
    for sess, session_data in enumerate(dataset):
        # WARNING train_batch is NOT a mini-batch, but one incremental batch!
        # You can later train with SGD indexing train_x and train_y properly.
        train_x, train_y, test_x, test_y = session_data
        imgs = trsf(train_x[0].astype(np.float32))
        print(imgs[:, 0, 0])
        print(imgs.size())
        sys.exit(0)

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
    
    
   
