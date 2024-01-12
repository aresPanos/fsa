# Directory for storing all the datasets used for the experiments
The datasets that should be stored in this directory are:
* CIFAR100
* CORE50
* SVHN
* Letters
* FGVC-Aircraft
* Stanford-cars
* i-Naturalist
* DomainNet

Letters, i-Naturalist, and DomainNet can be downloaded from this [Google-Drive](https://drive.google.com/drive/folders/105kYR9ZRbK_A9gD2alIn5wTUGYm3XiwR?usp=drive_link). The rest of the datasets are available from [torchvision datasets](https://pytorch.org/vision/0.16/datasets.html).
The tensorflow datasets such as `dSprites-xpos` can be downloaded using the [tensorflow_datasets (version=4.6.0)](https://www.tensorflow.org/datasets) and they should be stored in the directory `tensorflow_datasets`.
Finally, for the two datasets CIFAR100 and CUB200 of the `fscil` experiment (see Table 2 in the [paper](https://arxiv.org/pdf/2303.13199.pdf)), the exact splits are stored in the folder `fscil`.
