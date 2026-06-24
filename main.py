"""
=================================================
Main.py
----------

This is the main page for the code.
==================================================
"""


# 1. Imports
#----------------
import os
import yaml
import torch
import argparse
import utils.PATHS as PATHS



# 2. Creating the Parser for the arguments
#--------------------------------------------
def parse_args():
    """
    Function parses the arguments and returns the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None, help="Name of the model.")
    parser.add_argument("--dataset", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples used.")
    parser.add_argument("--seed", type=int, default=42, help="Seed number to be used throughout the whole experiment.")
    parser.add_argument("--output_dir", type=str, default="results", help="Path of the Output Directory where the results are stored.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Whether the user is on gpu or cpu.")

    return parser.parse_args()

# 3. Main Function
#--------------------
def main():
    args = parse_args()

    # loading the config.yaml file
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # assigning the arguments value between the config.yaml and the given arguments
    model_key = args.model if args.model is not None else config['default_model']
    model_name = config['models'][model_key]['name']
    model_path = config['models'][model_key]['path']

    dataset_key = args.dataset if args.dataset is not None else config['default_dataset']
    dataset_name = config['dataset'][dataset_key]['name']
    dataset_split = config['dataset'][dataset_key]['split']
    num_samples = args.num_samples if args.num_samples is not None else config['dataset'][dataset_key]['n_samples']

    seed = args.seed 

    output_dir = args.output_dir if args.output_dir is not None else config['paths']['output_dir']


    print(f"Model key:   {model_key}")
    print(f"Model name:  {model_name}")
    print(f"Model path:  {model_path}")
    print(f"Dataset:     {dataset_key}")
    print(f"Split:       {dataset_split}")
    print(f"Samples:     {num_samples}")
    print(f"Output dir:  {output_dir}")
    print(f"Device:      {args.device}")


if __name__ == "__main__":
    main()


