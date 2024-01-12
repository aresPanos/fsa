import numpy as np
import os
import torch
from dataloader.data_utils import *
from utils import limit_tensorflow_memory_usage

class Dataset_CL(object):

    def __init__(self, args):
        self.args = args
        self.session = 0
        dataset_shots_exp = CL_DATASETS[self.args.dataset_fs]

        if dataset_shots_exp['category'] != "rest":
            limit_tensorflow_memory_usage(1024)                                                              
            dataset_reader = TfDatasetReaderShots(
                    dataset=dataset_shots_exp['name'],
                    num_classes=dataset_shots_exp['num_classes'],
                    task=dataset_shots_exp['task'],
                    shots=self.args.train_shots,
                    target_size=2000,
                    path_to_datasets=os.path.join(args.datasets_path, "tensorflow_datasets"),
                    image_size=224,
                    seed=self.args.seed,
                    imagenet_norm=False,
            )
            train_images, train_labels, test_images, test_labels = dataset_reader._get_batch()
        else:
            if dataset_shots_exp['name'] in ["stanford_cars", "fgvc_aircraft"]:
                train_images, train_labels, test_images, test_labels, _ = cars_aircraft_dataset(self.args)
            elif dataset_shots_exp['name'] == "letters":
                train_images, train_labels, test_images, test_labels, _ = letters_dataset(self.args)
            else:
                train_images, train_labels, test_images, test_labels, _ = iNaturalist_domain_dataset(self.args)

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
    

    def __iter__(self):
        return self


    def __next__(self):
        """ Next batch based on the object parameter which can be also changed
            from the previous iteration. """ 

        if self.session == self.args.sessions:
            raise StopIteration

        # loading train-test data for this session
        train_images_session, train_labels_session, test_images_session, test_labels_session = self.extract_session_data()
        # Update state for next iter
        self.session += 1

        return (train_images_session, train_labels_session, test_images_session, test_labels_session)


    def extract_session_data(self):
        total_num_classes = self.args.base_class + self.session * self.args.way
        prev_num_classes = 0 if self.session == 0 else total_num_classes - self.args.way
        train_session_classes = np.arange(prev_num_classes, total_num_classes)
        test_session_classes = np.arange(total_num_classes)

        train_images_session, train_labels_session = [], []
        for cl in train_session_classes:
            idx_class = extract_class_indices(self.train_labels, cl)
            class_train_images = torch.index_select(self.train_images, 0, idx_class)
            class_train_labels = torch.index_select(self.train_labels, 0, idx_class)
            train_images_session.append(class_train_images)
            train_labels_session.append(class_train_labels)
        train_images_session = torch.vstack(train_images_session)
        train_labels_session = torch.cat(train_labels_session)

        test_images_session, test_labels_session = [], []
        for cl in test_session_classes:
            idx_class = extract_class_indices(self.test_labels, cl)
            class_test_images = torch.index_select(self.test_images, 0, idx_class)
            class_test_labels = torch.index_select(self.test_labels, 0, idx_class)
            test_images_session.append(class_test_images)
            test_labels_session.append(class_test_labels)
        test_images_session = torch.vstack(test_images_session)
        test_labels_session = torch.cat(test_labels_session)

        return (train_images_session, train_labels_session, test_images_session, test_labels_session)


    def reset_session(self):
        self.session = 0