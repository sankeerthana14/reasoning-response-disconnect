import os

THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(THIS_FILE)


# Parent Folder Directories
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')

