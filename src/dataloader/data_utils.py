import random
import os
import numpy as np
from argparse import Namespace

import torch
import torchvision.transforms as transforms
from torchvision.datasets import FGVCAircraft, StanfordCars
from torchvision.io import read_image

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image

CL_DATASETS = {"cifar100": {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
                    'num_classes': 100, 'enabled': True},
               "svhn": {'name': "svhn_cropped", 'task': None, 'model_name': "svhn", 'category': "natural",
                    'num_classes': 10, 'enabled': True},
               "dsprites-xpos": {'name': "dsprites", 'task': "location", 'model_name': "dsprites-xpos", 'category': "structured",
                    'num_classes': 16, 'enabled': True},
               "fgvc_aircraft": {'name': "fgvc_aircraft", 'task': None, 'model_name': "fgvc-aircraft", 'category': "rest",
                    'num_classes': 100, 'enabled': True},
               "stanford_cars": {'name': "stanford_cars", 'task': None, 'model_name': "stanford-cars", 'category': "rest",
                    'num_classes': 196, 'enabled': True},
               "letters": {'name': "letters", 'task': None, 'model_name': "letters", 'category': "rest",
                    'num_classes': 62, 'enabled': True},
               "domain_net": {'name': "domain_net", 'task': None, 'model_name': "domain_net", 'category': "rest",
                    'num_classes': 60, 'enabled': True},
               "i_naturalist": {'name': "i_naturalist", 'task': None, 'model_name': "i_naturalist", 'category': "rest",
                    'num_classes': 100, 'enabled': True}
}

IMG_SIZE = 224
MAX_IN_MEMORY = 20000

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class

    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def set_up_datasets(args: Namespace):
    if args.few_shots:
        if args.dataset_fs == 'cifar100':
            args.base_class = 20
            args.num_classes = 100
            args.way = 10
            args.sessions = 9

        if args.dataset_fs == 'svhn':
            args.base_class = 2
            args.num_classes = 10
            args.way = 2
            args.sessions = 5

        if args.dataset_fs == 'dsprites-xpos':
            args.base_class = 4
            args.num_classes = 16
            args.way = 2
            args.sessions = 7

        if args.dataset_fs == 'fgvc_aircraft':
            args.base_class = 20
            args.num_classes = 100
            args.way = 10
            args.sessions = 9

        if args.dataset_fs == 'stanford_cars':
            args.base_class = 36
            args.num_classes = 196
            args.way = 20
            args.sessions = 9

        if args.dataset_fs == 'letters':
            args.base_class = 12
            args.num_classes = 62
            args.way = 5
            args.sessions = 11

        if args.dataset_fs == 'domain_net':
            args.base_class = 12
            args.num_classes = 60
            args.way = 6
            args.sessions = 9

        if args.dataset_fs == 'i_naturalist':
            args.base_class = 20
            args.num_classes = 100
            args.way = 10
            args.sessions = 9

    elif args.fscil:
        if args.dataset_fscil == 'cifar100':
            import dataloader.cifar100_fscil as Dataset
            args.base_class = 60
            args.num_classes = 100
            args.way = 5
            args.shot = 5
            args.sessions = 9
    
        if args.dataset_fscil == 'cub200':
            import dataloader.cub200 as Dataset
            args.base_class = 100
            args.num_classes = 200
            args.way = 10
            args.shot = 5
            args.sessions = 11

        args.Dataset=Dataset
    else:
        if args.dataset == 'cifar100':
            import dataloader.cifar100 as Dataset
            args.base_class = 10
            args.num_classes = 100
            args.way = 10
            args.sessions = 10
    
        if args.dataset == 'core50':
            import dataloader.core50 as Dataset
            args.base_class = 10
            args.num_classes = 50
            args.way = 5
            args.sessions = 9

        if args.dataset == 'svhn':
            import dataloader.svhn as Dataset
            args.base_class = 2
            args.num_classes = 10
            args.way = 2
            args.sessions = 5

        if args.dataset == 'dsprites-xpos':
            import dataloader.dsprites as Dataset
            args.base_class = 4
            args.num_classes = 16
            args.way = 2
            args.sessions = 7

        if args.dataset == 'fgvc_aircraft':
            import dataloader.fgvc_aircraft as Dataset
            args.base_class = 10
            args.num_classes = 100
            args.way = 10
            args.sessions = 10

        if args.dataset == 'stanford_cars':
            import dataloader.stanford_cars as Dataset
            args.base_class = 16
            args.num_classes = 196
            args.way = 20
            args.sessions = 10

        if args.dataset == 'letters':
            import dataloader.letters as Dataset
            args.base_class = 12
            args.num_classes = 62
            args.way = 5
            args.sessions = 11
    
        args.Dataset=Dataset

    assert (args.base_class + (args.sessions-1) * args.way) == args.num_classes
    return args


def get_downloader_session(args: Namespace, session: int):
    class_index_all = np.arange(args.base_class + session * args.way)
    txt_path = os.path.join(args.datasets_path, "fscil", args.dataset_fscil, "session_1_%d.txt" %(session + 1))
    if args.dataset_fscil == 'cifar100':
        if session != 0:
            class_index = open(txt_path).read().splitlines()
            trainset = args.Dataset.CIFAR100(root=args.datasets_path, train=True, download=False, index=class_index, base_sess=False, 
                                             iscl_all=True, session=session, is_32_img_size=False)
        else:
            trainset = args.Dataset.CIFAR100(root=args.datasets_path, train=True, download=False, index=np.arange(args.base_class), base_sess=False, 
                                             iscl_all=True, session=session, is_32_img_size=False)
        testset = args.Dataset.CIFAR100(root=args.datasets_path, train=False, download=False, index=class_index_all, base_sess=False, 
                                        iscl_all=True, is_32_img_size=False)
    
    if args.dataset_fscil == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path, iscl_all=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index_all, iscl_all=True)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainloader, testloader


def get_cl_dataloader(args: Namespace):
    if args.few_shots:
        from dataloader.dataset_cl import Dataset_CL
        cl_loader = Dataset_CL(args)
    else:
        if args.dataset == 'cifar100':
            cl_loader = args.Dataset.CIFAR100(root=args.datasets_path)

        if args.dataset == 'core50':
            cl_loader = args.Dataset.CORE50(root=args.datasets_path)

        if args.dataset == 'svhn':
            cl_loader = args.Dataset.SVHN(root=args.datasets_path)

        if args.dataset == 'dsprites-xpos':
            cl_loader = args.Dataset.DSprites(root=args.datasets_path)

        if args.dataset == 'letters':
            cl_loader = args.Dataset.Letters(root=args.datasets_path)

        if args.dataset == 'fgvc_aircraft':
            cl_loader = args.Dataset.FGVCAircraft(root=args.datasets_path)
        
        if args.dataset == 'stanford_cars':
            cl_loader = args.Dataset.StanfordCars(root=args.datasets_path)

    return cl_loader


def get_session_classes(args: Namespace, session: int) -> np.ndarray:
    class_list = np.arange(args.base_class + session * args.way)
    return class_list


def get_tranforms(args: Namespace):
    mean_vec = [0.5, 0.5, 0.5]
    std_vec = [0.5, 0.5, 0.5]

    if (args.dataset == "cifar100" and (not args.few_shots)) or (args.dataset_fs == "cifar100" and args.few_shots):
        transforms_train = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean_vec, std=std_vec)
                        ])

        transforms_test = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=mean_vec, std=std_vec)
                          ])

    if (args.dataset == "core50" and (not args.few_shots)) or (args.dataset_fs == "core50" and args.few_shots):
        transforms_train = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean_vec, std=std_vec)
                           ])

        transforms_test = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=mean_vec, std=std_vec)
                          ])

    if (args.dataset == "svhn" and (not args.few_shots)) or (args.dataset_fs == "svhn" and args.few_shots):
        transforms_train = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=mean_vec, std=std_vec)
                          ])

        transforms_test = transforms_train

    if args.dataset_fs == "dsprites-xpos" and args.few_shots:
        transforms_train = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=mean_vec, std=std_vec)
                          ])

        transforms_test = transforms_train

    if args.dataset == "dsprites-xpos" and not args.few_shots:
        transforms_train = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Grayscale(num_output_channels=3),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=mean_vec, std=std_vec)
                          ])

        transforms_test = transforms_train

    if (args.dataset in ["fgvc_aircraft", "stanford_cars"] and (not args.few_shots)) or (args.dataset_fs in ["fgvc_aircraft", "stanford_cars"] and args.few_shots):
        transforms_train = transforms.Compose([
                           transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean_vec, std=std_vec)
                           ])

        transforms_test = transforms_train

    if args.dataset == 'letters' and not args.few_shots:
        transforms_train, transforms_test = None, None


    return transforms_train, transforms_test


class Dataset_transform(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.from_numpy(targets).type(torch.LongTensor)
        self.transform = transform  # save the transform

    def __len__(self):
        return len(self.targets)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def iNaturalist_domain_dataset(args):
    if args.dataset_fs == "i_naturalist":
        fname = "i_naturalist_shots=50.pt"
    else:
        fname = "domain_net_shots=50.pt"
        
    save_dir = os.path.join(args.datasets_path, args.dataset_fs, fname)
    data_dict = torch.load(save_dir)
    train_images, train_labels = data_dict["context_images"], data_dict["context_labels"]
    test_images, test_labels = data_dict["test_images"], data_dict["test_labels"]

    num_classes = 100 if args.dataset_fs == "i_naturalist" else 60
    assert num_classes == torch.unique(train_labels).size(0)

    return train_images, train_labels, test_images, test_labels, num_classes


def letters_dataset(args):
    num_classes = 62

    dir_letters = os.path.join(args.datasets_path, "letters", "English")
    dir_category = [["Img", "GoodImg", "Bmp"], ["Hnd", "Img"], ["Fnt"]]

    mean_norm, std_norm = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor()
                ])
    tranf_norm = transforms.Normalize(mean=mean_norm, std=std_norm)

    shots_per_categ = args.train_shots // 3
    left_shots = args.train_shots % 3

    np.random.seed(args.seed)
    choose_categ = np.random.choice(3, left_shots, replace=False)
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for id_categ, categ in enumerate(dir_category):
        images_per_class = np.zeros(num_classes, dtype=np.int64)
        use_extra_shot = id_categ in choose_categ
        for cl in range(num_classes):
            cl_name = "Sample00" + str(cl+1) if cl < 9 else "Sample0" + str(cl+1)
            path_class = os.path.join(dir_letters, *categ, cl_name)
            cls_num_imgs = len(os.listdir(path_class))
            train_cls_num_imgs = int(0.8*cls_num_imgs)
            images_per_class[cl] = cls_num_imgs

            random_perm = np.random.permutation(cls_num_imgs)
            if args.train_shots > 0:
                train_shots_class = shots_per_categ if train_cls_num_imgs > shots_per_categ else train_cls_num_imgs
                train_shots_class = train_shots_class + 1 if use_extra_shot else train_shots_class
            else:
                train_shots_class = train_cls_num_imgs

            train_split, test_split = random_perm[:train_shots_class], random_perm[train_cls_num_imgs:]

            for idx, f in enumerate(os.listdir(path_class)):
                img_path = os.path.join(path_class, f)
                if os.path.isfile(img_path):
                    if idx in train_split or idx in test_split:
                        img = read_image(img_path)
                        img = transf(img)
                        if img.size(0) == 1:
                            img = img.expand(3, *img.shape[1:])
                        img = tranf_norm(img)

                        if idx in train_split:
                            train_images.append(img)
                            train_labels.append(cl)
                        elif idx in test_split:
                            test_images.append(img)
                            test_labels.append(cl)
    
    train_images, train_labels = torch.stack(train_images), torch.LongTensor(train_labels)
    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = torch.stack(test_images), torch.LongTensor(test_labels)
    
    assert num_classes == torch.unique(train_labels).size(0)

    return train_images, train_labels, test_images, test_labels, num_classes


def cars_aircraft_dataset(args):
    mean_norm, std_norm = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transf = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_norm, std=std_norm)
                ])

    g = torch.Generator()
    g.manual_seed(args.seed + 12345)

    if args.dataset_fs == "stanford_cars":
        data_tr = StanfordCars(root=args.datasets_path, split="train", download=False, transform=transf)
        data_tst = StanfordCars(root=args.datasets_path, split="test", download=False, transform=transf)
    else:
        data_tr = FGVCAircraft(root=os.path.join(args.datasets_path, "fgvc_aircraft"), split="trainval", download=False, transform=transf)
        data_tst = FGVCAircraft(root=os.path.join(args.datasets_path, "fgvc_aircraft"), split="test", download=False, transform=transf)

    data_loader_tr = torch.utils.data.DataLoader(dataset=data_tr, batch_size=8200, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    data_loader_tst = torch.utils.data.DataLoader(dataset=data_tst, batch_size=8100, shuffle=False, num_workers=4)

    train_images, train_labels = next(iter(data_loader_tr))
    test_images, test_labels = next(iter(data_loader_tst))
    num_classes = torch.unique(train_labels).size(0)
    np.random.seed(1234 + args.seed)

    all_images_tr, all_labels_tr = [], []
    for cl_i in range(num_classes):
        train_idx = extract_class_indices(train_labels, cl_i)
        cls_train_images = torch.index_select(train_images, 0, train_idx)
        cls_train_labels = torch.index_select(train_labels, 0, train_idx)

        cls_train_images, cls_train_labels = shuffle(cls_train_images, cls_train_labels)
        if args.train_shots > 0:
            all_images_tr.append(cls_train_images[:args.train_shots])
            all_labels_tr.append(cls_train_labels[:args.train_shots])
        else:
            all_images_tr.append(cls_train_images)
            all_labels_tr.append(cls_train_labels)


    train_images, train_labels = torch.cat(all_images_tr, dim=0), torch.cat(all_labels_tr, dim=0)
    train_images, train_labels = shuffle(train_images, train_labels)
    
    return train_images, train_labels, test_images, test_labels, num_classes


class TfDatasetReaderShots:
    def __init__(self,
                 dataset: str,
                 num_classes: int,
                 task: str,
                 shots: int,
                 target_size: int,
                 path_to_datasets: str,
                 image_size: int=84,
                 seed: int=0,
                 imagenet_norm: bool=False,
                 upper_bound: int=500,
                ):
        self.dataset = dataset
        self.task = task
        self.image_size = image_size
        self.shots = shots
        self.test_shots = int(np.ceil(target_size / num_classes))
        self.num_classes = num_classes
        self.seed = seed
        tf.compat.v1.enable_eager_execution()

        if target_size < 2 * num_classes:
            ValueError('target_size value should be larger than two times the number of classes.')

        ds_context, ds_context_info = tfds.load(
            dataset,
            split='train',
            shuffle_files=False,
            data_dir=path_to_datasets,
            with_info=True,
            decoders=None
        )
        
        #print("Dataset: {} | Train images: {}" .format(dataset, ds_context_info.splits["train"].num_examples))
        if 'test' in ds_context_info.splits:
            if shots == -1:
                self.context_batch_size = min(ds_context_info.splits["train"].num_examples, 100_000)
                train_split = 'train'
            else:
                self.context_batch_size = min(ds_context_info.splits["train"].num_examples, upper_bound * self.num_classes)
        else:
            if shots == -1:
                self.context_batch_size = min(int(0.9 * ds_context_info.splits["train"].num_examples), 100_000)
            else:
                self.context_batch_size = min(ds_context_info.splits["train"].num_examples, upper_bound * self.num_classes)
                
        train_split = 'train[:{}]'.format(self.context_batch_size)
        decoders = None

        ds_context, ds_context_info = tfds.load(
            dataset,
            split=train_split,
            shuffle_files=True,
            data_dir=path_to_datasets,
            with_info=True,
            decoders=decoders,
            read_config=tfds.ReadConfig(shuffle_seed=self.seed)
        )

        self.context_dataset_length = ds_context_info.splits["train"].num_examples
        self.context_iterator = ds_context.as_numpy_iterator()
        
        test_split = 'test'
        if self.dataset == 'clevr':
            test_split = 'validation'
        if 'test' in ds_context_info.splits:
            # we use the entire test set
            ds_target, ds_target_info = tfds.load(
                dataset,
                split=test_split,
                shuffle_files=False,
                data_dir=path_to_datasets,
                with_info=True
            )
            self.target_dataset_length = ds_target_info.splits["test"].num_examples
            self.target_iterator = ds_target.as_numpy_iterator()
        else:  # there is no test split
            test_split = 'train[{}:]'.format(self.context_batch_size)
            ds_target, ds_target_info = tfds.load(
                dataset,
                split=test_split,
                shuffle_files=True,
                data_dir=path_to_datasets,
                with_info=True,
                decoders=None,
                read_config=tfds.ReadConfig(shuffle_seed=self.seed)
            )
            self.target_dataset_length = ds_target_info.splits[test_split].num_examples
            self.target_iterator = ds_target.as_numpy_iterator()
            #print('Test/Train num examples', self.target_dataset_length)
            #sys.exit(0)

        self.target_batch_size = min(self.target_dataset_length, 400 * self.num_classes)
        mean_vec = [0.485, 0.456, 0.406] if imagenet_norm else [0.5, 0.5, 0.5]
        std_vec = [0.229, 0.224, 0.225] if imagenet_norm else [0.5, 0.5, 0.5]
        normalize = transforms.Normalize(mean=mean_vec, std=std_vec)  # normalize to -1 to 1 or use Imagenet normalization 
        self.transforms = transforms.Compose([transforms.ToTensor(), normalize])


    def get_context_dataset_length(self):
        return self.context_dataset_length

    def get_target_dataset_length(self):
        return self.target_dataset_length

    def _get_batch(self):
        images_context = []
        labels_context = []
        for _ in range(self.context_batch_size):
            try:
                item = self.context_iterator.next()
            except StopIteration:  # the last batch may be less than batch_size
                break

            # images
            images_context.append(self._prepare_image(item['image']))

            # labels
            if self.dataset == "clevr":
                labels_context.append(self._get_clevr_label(item, self.task))
            elif self.dataset == 'kitti':
                labels_context.append(self._get_kitti_label(item))
            elif self.dataset == 'smallnorb':
                if self.task == 'azimuth':
                    labels_context.append(item['label_azimuth'])
                elif self.task == 'elevation':
                    labels_context.append(item['label_elevation'])
                else:
                    raise ValueError("Unsupported smallnorb task.")
            elif self.dataset == "dsprites":
                labels_context.append(self._get_dsprites_label(item, self.task))
            else:
                labels_context.append(item['label'])

        labels_context = np.array(labels_context)
        images_context = torch.stack(images_context)

        if self.shots != -1:
            _, class_counts = np.unique(labels_context, return_counts=True)

            np.random.seed(self.seed+123)
            indices = [idx
                    for c in range(self.num_classes)
                    for idx in np.random.choice(np.where(labels_context == c)[0],
                                                min(self.shots, class_counts[c]),
                                                replace=False)
                    ]

            images_context = images_context[indices]
            labels_context = labels_context[indices]

        images_target = []
        labels_target = []
        for _ in range(self.target_batch_size):
            try:
                item = self.target_iterator.next()
            except StopIteration:  # the last batch may be less than batch_size
                break

            # images
            images_target.append(self._prepare_image(item['image']))

            # labels
            if self.dataset == "clevr":
                labels_target.append(self._get_clevr_label(item, self.task))
            elif self.dataset == 'kitti':
                labels_target.append(self._get_kitti_label(item))
            elif self.dataset == 'smallnorb':
                if self.task == 'azimuth':
                    labels_target.append(item['label_azimuth'])
                elif self.task == 'elevation':
                    labels_target.append(item['label_elevation'])
                else:
                    raise ValueError("Unsupported smallnorb task.")
            elif self.dataset == "dsprites":
                labels_target.append(self._get_dsprites_label(item, self.task))
            else:
                labels_target.append(item['label'])

        labels_target = np.array(labels_target)
        images_target = torch.stack(images_target)

        _, class_counts = np.unique(labels_target, return_counts=True)

        np.random.seed(self.seed+1234)
        indices = [idx
                   for c in range(self.num_classes)
                   for idx in np.random.choice(np.where(labels_target == c)[0],
                                               min(self.test_shots, class_counts[c]),
                                               replace=False)
                   ]
        images_target = images_target[indices]
        labels_target = labels_target[indices]

        return images_context, torch.from_numpy(labels_context).type(torch.LongTensor), images_target, torch.from_numpy(labels_target).type(torch.LongTensor)


    def _get_kitti_label(self, x):
        """Predict the distance to the closest vehicle."""
        # Location feature contains (x, y, z) in meters w.r.t. the camera.
        vehicles = np.where(x["objects"]["type"] < 3)  # Car, Van, Truck.
        vehicle_z = np.take(x["objects"]["location"][:, 2], vehicles)
        if len(vehicle_z.shape) > 1:
            vehicle_z = np.squeeze(vehicle_z, axis=0)
        if vehicle_z.size == 0:
            vehicle_z = np.array([1000.0])
        else:
            vehicle_z = np.append(vehicle_z, [1000.0], axis=0)
        dist = np.amin(vehicle_z)
        # Results in a uniform distribution over three distances, plus one class for "no vehicle".
        thrs = np.array([-100.0, 8.0, 20.0, 999.0])
        label = np.amax(np.where((thrs - dist) < 0))
        return label

    def _get_dsprites_label(self, item, task):
        num_classes = 16
        if task == "location":
            predicted_attribute = 'label_x_position'
            num_original_classes = 32
        elif task == "orientation":
            predicted_attribute = 'label_orientation'
            num_original_classes = 40
        else:
            raise ValueError("Bad dsprites task.")

        # at the desired number of classes. This is useful for example for grouping
        # together different spatial positions.
        class_division_factor = float(num_original_classes) / float(num_classes)

        return np.floor(float(item[predicted_attribute]) / class_division_factor)

    def _get_clevr_label(self, item, task):
        if task == "count":
            label = len(item["objects"]["size"]) - 3
        elif task == "distance":
            dist = np.amin(item["objects"]["pixel_coords"][:, 2])
            # These thresholds are uniformly spaced and result in more or less balanced
            # distribution of classes, see the resulting histogram:
            thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
            label = np.amax(np.where((thrs - dist) < 0))
        else:
            raise ValueError("Bad clevr task.")

        return label

    def _prepare_image(self, image):
        if self.dataset == "smallnorb" or self.dataset == "dsprites":
            # grayscale images where the channel needs to be squeezed to keep PIL happy
            image = np.squeeze(image)

        if self.dataset == "dsprites":  # scale images to be in 0 - 255 range to keep PIL happy
            image = image * 255.0

        im = Image.fromarray(image)
        im = im.resize((self.image_size, self.image_size), Image.LANCZOS)
        im = im.convert("RGB")
        return self.transforms(im)


class Dataset_transform(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform  # save the transform

    def __len__(self):
        return len(self.targets)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class Dataset_transform_V2(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.transform_rgb = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                          ])

        self.transform_gray = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.Grayscale(num_output_channels=3),
                          transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                          ])

    def __len__(self):
        return len(self.targets)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = self.transform_rgb(img) if img.size(0) == 3 else self.transform_gray(img)
            
        return img, target