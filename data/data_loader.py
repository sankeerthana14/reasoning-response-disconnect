import os
import pandas as pd
from datasets import load_dataset

# pass in dataset_config['dataset']['truthfulqa']
def download_truthfulqa(dataset_config):

    tqa = load_dataset(dataset_config['hf_name'], dataset_config['hf_config'], split=dataset_config['split'])

    # correct_answers - possible other correct answers other than the GT answer
    rows = []

    for i, example in enumerate(tqa):
        rows.append({
            "id": f"truthfulqa_{i:05d}", # creating the question id
            "question": example['question'],
            "gt_answer": example['best_answer'],
            "correct_answers": example['correct_answers'],
            "incorrect_answers": example['incorrect_answers']
        })

    df = pd.DataFrame(rows)
    df.to_csv(dataset_config['raw_path'], index=False)
    print(f"INFO: TruthfulQA has been successfully saved as a CSV file at {dataset_config['raw_path']}!")


def download_simpleqa(dataset_config):
    sqa = pd.read_csv(dataset_config['url'])

    df = sqa[['problem', 'answer']].copy()
    df = df.rename(columns={'problem': 'question', 'answer': 'gt_answer'}) # rename the columns to match the other columns
    
    df['id'] = [f"simpleqa_{i:05d}" for i in range(len(df))]
    df = df[['id', 'question', 'gt_answer']]  # reorder columns
    df.to_csv(dataset_config['raw_path'], index=False)

    print(f"INFO: SimpleQA has been successfully saved as a CSV file at {dataset_config['raw_path']}!")


# config['dataset']['strategyqa']
def download_strategyqa(dataset_config):
    """
    This function downloads the strategyqa and extracts only the useful columns and saves them as a CSV file.
    """
    stratqa = load_dataset(dataset_config['hf_name'], split=dataset_config['split'])

    rows = []

    for i, example in enumerate(stratqa):
        rows.append({
            "id": f"strategyqa_{i:05d}",
            "question": example['question'],
            "gt_answer": str(example['answer']),
            "facts": example["facts"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(dataset_config['raw_path'], index=False)
    print(f"INFO: StrategyQA has been successfully saved as a CSV file at {dataset_config['raw_path']}!")


def load_data(dataset_key, dataset_config, num_samples, seed):
    raw_path = dataset_config['raw_path']
    
    # Download and format the data if not already downloaded
    if not os.path.exists(raw_path):
        # download, normalise, save, return
        if dataset_key == "truthfulqa":
            download_truthfulqa(dataset_config)
        elif dataset_key == "simpleqa":
            download_simpleqa(dataset_config)
        elif dataset_key == "strategyqa":
            download_strategyqa(dataset_config)

    # Loading the downloaded data
    df = pd.read_csv(raw_path)

    # If num_samples that is not None is given:
    if num_samples is not None:
        df = df.sample(num_samples, random_state=seed).reset_index(drop=True)

    return df


        