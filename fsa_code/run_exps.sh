#!/bin/sh

datasets_cl="cifar100 svhn dsprites-xpos fgvc_aircraft letters domain_net i_naturalist"
seedslist="0 1 2 3 4"
for sd in $seedslist
do
    for ds in $datasets_cl
    do
        sbatch /home/ap2313/rds/hpc-work/slurm_jobs/run_job.sh "/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/train.py --project fenc --shots_exp --dataset_exp $ds --train_shots 50 --epochs_base 200 --batch_size 256 --base_lr 0.0001 --seed $sd >>/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/$ds-FENC-EffnetB0_exp_shots=50_seed=$sd.txt"
        sbatch /home/ap2313/rds/hpc-work/slurm_jobs/run_job.sh "/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/train.py --project fenc_film --shots_exp --dataset_exp $ds --train_shots 50 --epochs_base 200 --batch_size 256 --base_lr 0.005 --seed $sd >>/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/$ds-FENC_FiLM-EffnetB0_exp_shots=50_seed=$sd.txt"
    
    done
    sbatch /home/ap2313/rds/hpc-work/slurm_jobs/run_job.sh "/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/train.py --project fenc --dataset $ds --batch_size 256 --epochs_base 200 --base_lr 0.005 >>/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/$ds-full_shots-FENC_right_batch_size.txt"
    sbatch /home/ap2313/rds/hpc-work/slurm_jobs/run_job.sh "/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/train.py --project fenc_film --dataset $ds --batch_size 256 --epochs_base 200 --base_lr 0.005 >>/home/ap2313/rds/hpc-work/code/cifar100_core50_CL/$ds-full_shots-FENC_FiLM_right_batch_size.txt"
done

