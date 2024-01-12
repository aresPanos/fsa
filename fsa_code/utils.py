import os
import time
import random
from argparse import Namespace
from typing import Tuple

import numpy as np

import torch
from torch import Tensor

    
def get_dir_save(args: Namespace) -> str:
    if args.few_shots:
        save_path = os.path.join('few_shots',  args.dataset_fs)
    elif args.fscil:
        save_path = os.path.join('fscil',  args.dataset_fscil)
    else:
        save_path = os.path.join('full_shots',  args.dataset)

    str_film = "FiLM" if args.use_film else "FullBody"
    save_path = os.path.join(save_path,  str_film)
    save_path_tmp = 'Ftx_efficientnet-b0-Epo_%d-Lr_%.4f' % (args.epochs_base, args.base_lr)
    if args.few_shots:
        save_path_tmp += '-shots_%d-seed_%d' % (args.train_shots, args.seed)

    save_path = os.path.join(args.results_dir, save_path, save_path_tmp)
    ensure_path(save_path)
    
    return save_path


def get_batch_indices(index, last_element, batch_size):
    batch_start_index = index * batch_size
    batch_end_index = batch_start_index + batch_size
    if batch_end_index > last_element:
        batch_end_index = last_element
    return batch_start_index, batch_end_index


def extract_class_indices(labels, which_class) -> Tensor:
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def shuffle(images, labels):
    """
    Return shuffled data.
    """
    perm = np.random.permutation(len(labels))
    if isinstance(images, list):
        img_perm = [images[idx] for idx in perm]
        return img_perm, labels[perm]
    else:
        return images[perm], labels[perm]
    

def set_seed(seed: int):
    if seed < 0:
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path: str):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)
        
def estimate_cov(examples) -> Tensor:
    if examples.size(0) > 1:
        return torch.cov(examples.t(), correction=1)
    else:
        return torch.cov(examples.t(), correction=0)


def extract_class_indices(labels, which_class) -> Tensor:
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def mean_pooling(x) -> Tensor:
    return torch.mean(x, dim=0, keepdim=True)


def get_batch_indices(index: int, last_element: int, batch_size: int) -> Tuple[int, int, int]:
    batch_start_index = index * batch_size
    batch_end_index = batch_start_index + batch_size
    if batch_end_index > last_element:
        batch_end_index = last_element
    return batch_start_index, batch_end_index
    

def compute_lda_head(features, labels, args) -> Tuple[Tensor, Tensor]:
    cov_est = estimate_cov(features) + torch.eye(features.size(1)).to(args.device)
    chol = torch.linalg.cholesky(cov_est)

    class_means = torch.zeros((args.num_classes, features.size(1)), device=args.device)
    for c in torch.unique(labels):
        # filter out feature vectors which have class c
        class_features = torch.index_select(features, 0, extract_class_indices(labels, c))
        class_means[c.item()] = mean_pooling(class_features)

    class_means.t_()
    coeff = torch.cholesky_solve(class_means, chol) # output_size x num_classes
    bias = (coeff * class_means).sum(dim=0) # num_classes x None

    return coeff, bias


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def predict_by_max_logit(logits):
    return torch.argmax(logits, dim=-1)


def compute_accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    return compute_accuracy_from_predictions(labels, predict_by_max_logit(logits))


def compute_accuracy_from_predictions(predictions, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, predictions).float())

    
def count_acc(logits, label) -> float:
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def limit_tensorflow_memory_usage(gpu_memory_limit):
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()
    import tensorflow as tf
    
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


