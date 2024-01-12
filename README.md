# First Session Adaptation model
This repository contains the Pytorch implementation of the FSA/FSA-FiLM method from the ICCV-23 paper [First Session Adaptation: A Strong Replay-Free Baseline for
Class-Incremental Learning](https://arxiv.org/pdf/2303.13199.pdf). Please cite the following bib entry in case you use this code repository:

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

## Train and Evaluate FSA

To train and evaluate the FSA model on CIFAR100 dataset, run

    python fsa_code/train_eval.py --dataset cifar100 --datasets_path <directory_of_datasets> --results_dir ./results

The learned parameters of the mixture of log-Normals are stored at  `<directory_of_logs>/taxi/time_dist/saved_models/model_numMixtures-2.pt`

Next, to train and evaluate the mark model using D=128 and L=2 layers for the Transformer architecture, run

    python code/run/train_eval_mark_dist.py --dataset taxi -dmodel 64 -nLayers 2 --data_dir <same_as_above> --log_dir <same_as_above>

The fine-tuned parameters of the model are stored at  `./results/<task>/cifar100//saved_models/XFMRNHPFast_dmodel-64_nLayers-2_nHeads-2_date-time.pt`
where task $\in$ {`high_shots`, `fscil`, `few_shots`}.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/aresPanos/dtpp/blob/main/LICENSE) file for details.

## Acknowledgements
Our code is based on the following repositories:
* [Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)
* [HYPRO](https://github.com/ant-research/hypro_tpp/tree/main)

If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):



