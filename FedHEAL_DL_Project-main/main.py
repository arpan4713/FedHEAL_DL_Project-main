import os
import sys
import socket
import torch.multiprocessing
import logging
import uuid
import datetime
import warnings
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models, get_model
from utils.args import add_management_args
from datasets import get_prive_dataset
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from skopt import BayesSearchCV  # For hyperparameter tuning

# Set multiprocessing strategy and ignore warnings
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

# Set paths
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

# Initialize logger
LOG_FILE = os.path.join(conf_path, "training_metrics.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(conf_path, 'logs'))

# Adaptive learning rate function
def adaptive_lr(optimizer, client_metrics, threshold=0.1):
    """
    Adjust learning rate based on client's loss variability.
    """
    for idx, client_metric in enumerate(client_metrics):
        avg_loss = sum(client_metric['loss']) / len(client_metric['loss'])
        if avg_loss > threshold:  # Decrease lr if loss is high
            for param_group in optimizer[idx].param_groups:
                param_group['lr'] *= 0.5
        else:  # Increase lr for better convergence
            for param_group in optimizer[idx].param_groups:
                param_group['lr'] *= 1.1

# Apply dropout during training
def apply_dropout(model, dropout_rate=0.3):
    """
    Apply dropout to the model's layers.
    """
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout_rate

def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)

    # Define all the arguments you're using in the command line
    parser.add_argument('--device_id', type=int, help='Device ID for training')
    parser.add_argument('--communication_epoch', type=int, default=200, help='Number of communication epochs')
    parser.add_argument('--local_epoch', type=int, default=10, help='Number of local epochs')
    parser.add_argument('--parti_num', type=int, default=20, help='Number of participants')
    parser.add_argument('--model', type=str, default='fedavgheal', choices=['fedavgheal', 'other_models'], help='Model to use')
    parser.add_argument('--dataset', type=str, default='fl_digits', choices=['fl_digits', 'other_datasets'], help='Dataset to use')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter')
    parser.add_argument('--online_ratio', type=float, default=1.0, help='Online ratio')
    parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay')
    parser.add_argument('--averaging', type=str, default='weight', choices=['weight', 'other_averaging_methods'], help='Averaging method')
    parser.add_argument('--wHEAL', type=int, default=1, help='Weight HEAL parameter')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for some operation')
    parser.add_argument('--beta', type=float, default=0.4, help='Beta parameter')
    parser.add_argument('--structure', type=str, required=True, help='Structure of the model')
    parser.add_argument('--csv_log', action='store_true', help='Enable CSV logging for metrics')
    parser.add_argument('--rand_dataset', action='store_true', help='Enable random dataset')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--test_only', action='store_true', help='Flag for testing phase')

    # Parse the arguments
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]
    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)
    return args

def log_metrics(metrics, epoch):
    """
    Log the metrics to a file and TensorBoard.
    """
    log_message = f"Epoch: {epoch}, Metrics: {metrics}"
    logging.info(log_message)
    writer.add_scalars('Metrics', metrics, epoch)

def test_model(model, priv_dataset, args):
    """
    Testing phase for the model.
    """
    model.eval()  # Set the model to evaluation mode
    metrics = {'accuracy': 0.0}  # Placeholder for metrics calculation, modify as needed
    # Add testing code here based on your dataset and model, typically this includes:
    # - Iterating over the test dataset
    # - Evaluating the model's performance (e.g., accuracy, loss)
    print("Testing model...")
    return metrics

def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # Dataset and model initialization
    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, None)
    model = get_model(backbones_list, args, priv_dataset.get_transform())

    # Apply dropout during training
    apply_dropout(model, dropout_rate=0.3)

    # Custom model initialization
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    args.arch = model.nets_list[0].name

    print(f"Model: {args.model}, Part: {args.parti_num}, Dataset: {args.dataset}, "
          f"Comm Epoch: {args.communication_epoch}, Local Epoch: {args.local_epoch}")

    if args.test_only:
        # Testing phase
        metrics = test_model(model, priv_dataset, args)
        log_metrics(metrics, 0)  # Log testing metrics
    else:
        # Training phase
        metrics = train(model, priv_dataset, args)
        log_metrics(metrics, args.communication_epoch)

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()


