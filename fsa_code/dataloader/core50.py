#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 23-07-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Data Loader for the CORe50 Dataset """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# other imports
import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image


class CORE50(object):
    """ CORe50 Data Loader calss

    Args:
        root (string): Root directory of the dataset where ``core50_128x128``,
            ``paths.pkl``, ``LUP.pkl``, ``labels.pkl``, ``core50_imgs.npz``
            live. For example ``~/data/core50``.
        preload (string, optional): If True data is pre-loaded with look-up
            tables. RAM usage may be high.
        scenario (string, optional): One of the three scenarios of the CORe50
            benchmark ``ni``, ``nc``, ``nic``, `nicv2_79`,``nicv2_196`` and
             ``nicv2_391``.
        run (int, optional): One of the 10 runs (from 0 to 9) in which the
            training batch order is changed as in the official benchmark.
        start_batch (int, optional): One of the training incremental batches
            from 0 to max-batch - 1. Remember that for the ``ni``, ``nc`` and
            ``nic`` we have respectively 8, 9 and 79 incremental batches. If
            ``train=False`` this parameter will be ignored.
    """

    nsessions = 9
 
    def __init__(self, root, run=0):
        """" Initialize Object """

        self.root = os.path.join(root, "core50")
        self.preload = True
        self.run = run
        self.session = 0

        bin_path = os.path.join(self.root, 'core50_imgs.bin')
        if os.path.exists(bin_path):
            with open(bin_path, 'rb') as f:
                self.x = np.fromfile(f, dtype=np.uint8).reshape(164866, 128, 128, 3)

        else:
            with open(os.path.join(self.root, 'core50_imgs.npz'), 'rb') as f:
                npzfile = np.load(f)
                self.x = npzfile['x']
                self.x.tofile(bin_path)

        with open(os.path.join(self.root, 'paths.pkl'), 'rb') as f:
            self.paths = pkl.load(f)

        with open(os.path.join(self.root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        with open(os.path.join(self.root, 'labels.pkl'), 'rb') as f:
            self.labels = pkl.load(f)

        self.data_test, self.targets_test = self.get_test_set()


    def __iter__(self):
        return self

    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.nsessions:
            raise StopIteration

        # Getting the right indexis
        train_idx_list = self.LUP["nc"][self.run][self.session]

        # loading train data
        train_x = np.take(self.x, train_idx_list, axis=0)       

        train_y = self.labels["nc"][self.run][self.session]
        train_y = np.asarray(train_y)

        # loading test data
        index = np.arange(10) if self.session == 0 else np.arange(10 + self.session * 5)
        test_x, test_y = self.SelectfromDefault(self.data_test, self.targets_test, index)
        # Update state for next iter
        self.session += 1

        return (train_x, train_y, test_x, test_y)


    def get_test_set(self):
        """ Return the test set (the same for each inc. session). """

        test_idx_list = self.LUP["nc"][self.run][-1]
        test_x = np.take(self.x, test_idx_list, axis=0)

        test_y = self.labels["nc"][self.run][-1]
        test_y = np.asarray(test_y)

        return test_x, test_y

    
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


if __name__ == "__main__":
    # Create the dataset object for example with the "NIC_v2 - 79 benchmark"
    # and assuming the core50 location in ~/core50/128x128/
    dataset = CORE50(root="/scratch3/ap2313/datasets/core50/")

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
