# import os
# import sys
# import socket
# import torch.multiprocessing
# import logging
# import uuid
# import datetime
# import warnings
# from argparse import ArgumentParser
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from skopt import BayesSearchCV  # For hyperparameter tuning
# from torch.utils.tensorboard import SummaryWriter
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models, get_model
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# # Set multiprocessing strategy and ignore warnings
# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Paths setup
# conf_path = os.getcwd()
# sys.path.extend([conf_path, os.path.join(conf_path, 'datasets'), os.path.join(conf_path, 'backbone'), os.path.join(conf_path, 'models')])

# # Logger configuration
# LOG_FILE = os.path.join(conf_path, "training_metrics.log")
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # TensorBoard writer
# writer = SummaryWriter(log_dir=os.path.join(conf_path, 'logs'))

# # Pareto Front class for multi-objective optimization
# class ParetoFront:
#     def __init__(self):
#         self.solutions = []

#     def is_dominated(self, new_solution):
#         return any(sol[0] >= new_solution[0] and sol[1] <= new_solution[1] for sol in self.solutions)

#     def add_solution(self, new_solution):
#         if not self.is_dominated(new_solution):
#             self.solutions.append(new_solution)
#             self.solutions.sort(key=lambda x: x[0])  # Sort by accuracy

#     def get_solutions(self):
#         return self.solutions

# # Evaluation functions
# def evaluate_model(model, priv_dataset):
#     return model.evaluate(priv_dataset.test_data)

# def calculate_communication_cost(model, priv_dataset, args):
#     total_size = sum(param.numel() for param in model.parameters())
#     return total_size * len(priv_dataset.train_data)  # Proxy metric

# def evaluate_objectives(model, priv_dataset, args):
#     accuracy = evaluate_model(model, priv_dataset)
#     comm_cost = calculate_communication_cost(model, priv_dataset, args)
#     return accuracy, comm_cost

# # Adaptive learning rate function
# def adaptive_lr(optimizer, client_metrics, threshold=0.1):
#     for idx, client_metric in enumerate(client_metrics):
#         avg_loss = sum(client_metric['loss']) / len(client_metric['loss'])
#         for param_group in optimizer[idx].param_groups:
#             param_group['lr'] *= 0.5 if avg_loss > threshold else 1.1

# # Utility functions
# def apply_dropout(model, dropout_rate=0.3):
#     for module in model.modules():
#         if isinstance(module, torch.nn.Dropout):
#             module.p = dropout_rate

# def log_metrics(metrics, epoch):
#     logging.info(f"Epoch: {epoch}, Metrics: {metrics}")
#     writer.add_scalars('Metrics', metrics, epoch)

# def log_pareto_front(pareto_front):
#     for idx, solution in enumerate(pareto_front.get_solutions()):
#         logging.info(f"Solution {idx+1}: Accuracy = {solution[0]}, Communication Cost = {solution[1]}")

# def plot_pareto_front(pareto_front):
#     accuracies = [solution[0] for solution in pareto_front.get_solutions()]
#     comm_costs = [solution[1] for solution in pareto_front.get_solutions()]
#     plt.scatter(comm_costs, accuracies, color='b')
#     plt.title("Pareto Front: Accuracy vs Communication Cost")
#     plt.xlabel("Communication Cost (Lower is Better)")
#     plt.ylabel("Accuracy (Higher is Better)")
#     plt.grid(True)
#     plt.show()

# # Argument parsing
# def parse_args():
#     parser = ArgumentParser(description='Federated Learning Framework', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='Device ID')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='Communication Epochs')
#     parser.add_argument('--local_epoch', type=int, default=10, help='Local Training Epochs')
#     parser.add_argument('--parti_num', type=int, default=20, help='Number of Participants')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models())
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES)
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
#     parser.add_argument('--rand_dataset', type=int, default=1, help='Randomized dataset flag.')
#     parser.add_argument('--structure', type=str, default='homogeneity', help='Data distribution structure.')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID data.')
#     parser.add_argument('--online_ratio', type=float, default=1.0, help='Ratio of online clients.')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option.')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy.')
#     parser.add_argument('--wHEAL', type=int, default=1, help='Enable Weighted HEAL.')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for HEAL.')
#     parser.add_argument('--beta', type=float, default=0.4, help='Beta for momentum updates.')
#     add_management_args(parser)
#     args = parser.parse_args()

#     if args.dataset in best_args and args.model in best_args[args.dataset]:
#         for key, value in best_args[args.dataset][args.model].items():
#             setattr(args, key, value)

#     set_random_seed(args.seed)
#     return args

# # Main function
# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())

#     apply_dropout(model, dropout_rate=0.3)
#     model.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) else None)
    
#     pareto_front = ParetoFront()
#     for epoch in range(args.communication_epoch):
#         metrics = train(model, priv_dataset, args)
#         accuracy, comm_cost = evaluate_objectives(model, priv_dataset, args)
#         pareto_front.add_solution((accuracy, comm_cost))
#         log_metrics(metrics, epoch)

#         if epoch % 10 == 0:
#             log_pareto_front(pareto_front)

#     log_pareto_front(pareto_front)
#     plot_pareto_front(pareto_front)
#     writer.close()

# if __name__ == '__main__':
#     main()



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
# from sklearn.cluster import KMeans
# from torch.utils.tensorboard import SummaryWriter
# from skopt import BayesSearchCV  # For hyperparameter tuning


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


# # TensorBoard writer
# writer = SummaryWriter(log_dir=os.path.join(conf_path, 'logs'))


# # Adaptive learning rate function
# def adaptive_lr(optimizer, client_metrics, threshold=0.1):
#     """
#     Adjust learning rate based on client's loss variability.
#     """
#     for idx, client_metric in enumerate(client_metrics):
#         avg_loss = sum(client_metric['loss']) / len(client_metric['loss'])
#         if avg_loss > threshold:  # Decrease lr if loss is high
#             for param_group in optimizer[idx].param_groups:
#                 param_group['lr'] *= 0.5
#         else:  # Increase lr for better convergence
#             for param_group in optimizer[idx].param_groups:
#                 param_group['lr'] *= 1.1


# # Additional utility functions for enhancements (e.g., dynamic averaging, dropout, etc.)
# # Add the additional functions here (dynamic_averaging, apply_dropout, fedprox_aggregation, etc.)
# def apply_dropout(model, dropout_rate=0.3):
#     """
#     Apply dropout to all layers in the model that support it.
#     """
#     for layer in model.modules():
#         if isinstance(layer, torch.nn.Dropout):
#             layer.p = dropout_rate


# def parse_args():
#     parser = ArgumentParser(description='Federated Learning Framework', allow_abbrev=False)

#     # Argument definitions
#     parser.add_argument('--device_id', type=int, default=0, help='Device ID')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='Communication Epochs')
#     parser.add_argument('--local_epoch', type=int, default=10, help='Local Training Epochs')
#     parser.add_argument('--parti_num', type=int, default=20, help='Number of Participants')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model type')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset name')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter')
#     parser.add_argument('--online_ratio', type=float, default=1.0, help='Online ratio')
#     parser.add_argument('--learning_decay', type=float, default=0, help='Learning decay')
#     parser.add_argument('--averaging', type=str, default='weight', choices=['weight', 'other'], help='Averaging method')
#     parser.add_argument('--wHEAL', type=int, default=1, help='wHEAL flag')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold value')
#     parser.add_argument('--beta', type=float, default=0.4, help='Beta parameter')

#     # Management args (add any additional management-related arguments)
#     add_management_args(parser)  # Assuming this function adds other necessary arguments

#     # Parse arguments
#     args = parser.parse_args()

#     # Best args initialization based on dataset and model
#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     # Set random seed if provided
#     if args.seed is not None:
#         set_random_seed(args.seed)

#     return args


# def log_metrics(metrics, epoch):
#     """
#     Log the metrics to a file and TensorBoard.
#     """
#     log_message = f"Epoch: {epoch}, Metrics: {metrics}"
#     logging.info(log_message)
#     writer.add_scalars('Metrics', metrics, epoch)


# def main(args=None):
#     if args is None:
#         args = parse_args()

#     # Set unique job and timestamp details
#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     # Dataset and model initialization
#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
   
#     # Apply dropout during training
#     apply_dropout(model, dropout_rate=0.3)
   
#     # Custom model initialization (e.g., Xavier initialization)
#     def init_weights(m):
#         if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
#             torch.nn.init.xavier_uniform_(m.weight)
#     model.apply(init_weights)
   
#     # Store model architecture
#     args.arch = model.nets_list[0].name

#     print(f"Model: {args.model}, Part: {args.parti_num}, Dataset: {args.dataset}, "
#           f"Comm Epoch: {args.communication_epoch}, Local Epoch: {args.local_epoch}")

#     # Train model and get metrics
#     metrics = train(model, priv_dataset, args)

#     # Log final metrics to file and TensorBoard
#     log_metrics(metrics, args.communication_epoch)

#     # Close TensorBoard writer
#     writer.close()


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

def apply_dropout(model, dropout_rate=0.3):
    """
    Apply dropout to the model's layers.
    """
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout_rate

def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    # Add arguments here (same as before)
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
