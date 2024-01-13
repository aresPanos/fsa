# First Session Adaptation: A Strong Replay-Free Baseline forClass-Incremental Learning
This repository contains the Pytorch implementation of the FSA/FSA-FiLM method from the ICCV-23 paper [First Session Adaptation: A Strong Replay-Free Baseline for Class-Incremental Learning](https://arxiv.org/pdf/2303.13199.pdf). Please cite the following bib entry in case you use this code repository:

    @article{panos2023first,
        title={First Session Adaptation: A Strong Replay-Free Baseline for Class-Incremental Learning},
        author={Panos, Aristeidis and Kobe, Yuriko and Reino, Daniel Olmeda and Aljundi, Rahaf and Turner, Richard E},
        journal={arXiv preprint arXiv:2303.13199},
        year={2023}
    }

## Environment Requirements

Ensure that all the dependencies have been installed using the `requirements.txt` file.

## Data preparation

See [here](https://github.com/aresPanos/fsa/tree/main/datasets) for instructions regarding the datasets required for the experiments.

## GPU Requirements
The experiments in the paper run on a single NVIDIA A100 GPU with 80 GB of memory. For GPUs with smaller memory, one could use smaller batch size but this could potentially give slightly different results.

## Train and Evaluate FSA/FSA-FiLM

The default pretrained feature extractor used for the majority of the experiments in the paper is the <b>EfficientNet-B0</b>. The pretrained weights (ImageNet-1k) are downloaded automatically in case they are not found stored in pytorch's cache directory.
To train and evaluate the FSA model on CIFAR100 dataset for the high-shot setting, run

    python src/train_eval.py --dataset cifar100 --datasets_path <directory_of_datasets> --results_dir ./results

Similarly, for fine-tuning only the FiLM parameters use the flag `use_film` and run

    python src/train_eval.py --use_film --dataset cifar100 --datasets_path <directory_of_datasets> --results_dir ./results

For the FSCIL experiment of Table 2 in the paper, run 

    python src/train_eval.py --use_film --fscil --dataset_fscil cifar100 --datasets_path <directory_of_datasets> --results_dir ./results

Finally, for the few-shot setting using 50 shots run

    python src/train_eval.py --use_film --few_shots --train_shots 50 --dataset_fs cifar100 --datasets_path <directory_of_datasets> --results_dir ./results 
    
The fine-tuned parameters of the model are stored at `./results/<task>/cifar100/<method>/saved_models/<model_name>.pt` where task $\in$ {`high_shots`, `fscil`, `few_shots`} and method=`FullBody` for FSA or method=`FiLM` for FSA-FiLM.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/aresPanos/fsa/blob/main/LICENSE) file for details.

## Acknowledgements
Our code is based on the following repositories:
* [FiT](https://github.com/cambridge-mlg/fit/tree/main)
* [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
