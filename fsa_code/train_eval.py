import argparse
import sys
import datetime

from loguru import logger

from utils import *
from methods.cil_trainer import CIL_Trainer

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # about dataset and method
    parser.add_argument('--use_film', action='store_true', default=False, help='Use FSA or FSA-FILM method.')
    parser.add_argument('--results_dir', type=str, required=True, help="Directory where the results are stored.")
    parser.add_argument("--datasets_path",  required=True, help="Directory of the datasets.")

    # Full-shots datasets
    parser.add_argument("--dataset", type=str, default="cifar100", 
                        choices=["core50", "cifar100", "svhn", "letters", "dsprites-xpos", "fgvc_aircraft", "stanford_cars"])

    # Few-shots datasets
    parser.add_argument('--few_shots', action='store_true', default=False, help='Few shots experiment.')
    parser.add_argument('--dataset_fs', type=str, default="cifar100",
                        choices=["cifar100", "svhn", "dsprites-xpos", "fgvc_aircraft", "letters", "stanford_cars", "i_naturalist", "domain_net"])
    parser.add_argument("--train_shots", type=int, default=50, help="Images per class during training for few_shots.")

    # FSCIL datasets
    parser.add_argument('--fscil', action='store_true', default=False, help='FSCIL experiment.')
    parser.add_argument("--dataset_fscil", type=str, default="cifar100", choices=["cifar100", "cub200"])

    # about training
    parser.add_argument("--epochs_base", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)    
    
    return parser


if __name__ == "__main__":
    parser = get_command_line_parser()
    args = parser.parse_args()
    
    if args.few_shots: 
        assert not args.fscil, "The value of argument `fscil` should be False."
    if args.fscil: 
        assert not args.few_shots, "The value of argument `few_shots` should be False."
    if not args.few_shots and not args.fscil: 
        assert os.path.exists(args.datasets_path), "The value of argument `datasets_path` should be a valid directory."
    
    args.dir_save = get_dir_save(args)
    date_format = "%Y%m%d-%H%M%S"
    now_timestamp = datetime.datetime.now().strftime(date_format)
        
    logger.add(os.path.join(args.dir_save, f"log_{now_timestamp}.txt"))
    logger.info(args)   
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = CIL_Trainer(args, logger)        
    trainer.train_eval()
