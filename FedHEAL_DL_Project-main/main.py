# import os
# import sys
# import socket
# import torch.multiprocessing
# import logging
# import uuid
# import datetime
# import warnings
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models, get_model
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed
# from argparse import ArgumentParser


# # Set multiprocessing strategy and ignore warnings
# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Set paths
# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

# # Initialize logger
# LOG_FILE = os.path.join(conf_path, "training_metrics.log")
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
#     parser.add_argument('--seed', type=int, default=0, help='The random seed.')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='The random seed.')

#     # Add model and dataset options
#     parser.add_argument('--model', type=str, default='fedavgheal', 
#                         help='Model name.', choices=get_all_models())
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', 
#                         choices=DATASET_NAMES, help='Which scenario to perform experiments on.')
#     parser.add_argument('--alpha', type=float, default=0.5, help='alpha of dirichlet sampler.')
#     parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')
#     parser.add_argument('--learning_decay', type=int, default=0, help='The Option for Learning Rate Decay')
#     parser.add_argument('--averaging', type=str, default='weight', help='The Option for averaging strategy')

#     parser.add_argument('--wHEAL', type=int, default=1, help='The CORE of the FedHEAL decides whether to add HEAL to other FL method')
#     parser.add_argument('--threshold', type=float, default=0.3, help='threshold of HEAL')
#     parser.add_argument('--beta', type=float, default=0.4, help='momentum update beta')
    
#     parser.add_argument('--mnist', type=int, default=5, help='Number of mnist clients')
#     parser.add_argument('--usps', type=int, default=5, help='Number of usps clients')
#     parser.add_argument('--svhn', type=int, default=5, help='Number of svhn clients')
#     parser.add_argument('--syn', type=int, default=5, help='Number of syn clients')
    
#     parser.add_argument('--caltech', type=int, default=5, help='Number of caltech clients')
#     parser.add_argument('--amazon', type=int, default=5, help='Number of amazon clients')
#     parser.add_argument('--webcam', type=int, default=5, help='Number of webcam clients')
#     parser.add_argument('--dslr', type=int, default=5, help='Number of dslr clients')
    
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]

#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args


# def log_metrics(metrics, epoch):
#     """
#     Log the metrics to a file.
#     """
#     log_message = f"Epoch: {epoch}, Metrics: {metrics}"
#     logging.info(log_message)


# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     # Dataset and model initialization
#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
    
#     args.arch = model.nets_list[0].name

#     print(f"Model: {args.model}, Part: {args.parti_num}, Dataset: {args.dataset}, "
#           f"Comm Epoch: {args.communication_epoch}, Local Epoch: {args.local_epoch}")

#     # Train model and get metrics
#     metrics = train(model, priv_dataset, args)

#     # Log final metrics to file
#     log_metrics(metrics, args.communication_epoch)


# if __name__ == '__main__':
#     main()



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

# Additional utility functions for enhancements (e.g., dynamic averaging, dropout, etc.)
# Add the additional functions here (dynamic_averaging, apply_dropout, fedprox_aggregation, etc.)

def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    # (Same as before)
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

    # Train model and get metrics
    metrics = train(model, priv_dataset, args)

    # Log final metrics to file and TensorBoard
    log_metrics(metrics, args.communication_epoch)

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()

