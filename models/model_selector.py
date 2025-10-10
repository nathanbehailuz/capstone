# import torch
# import os

# from models.GNN_paper import NodeGCN as GNN_NodeGCN
# from models.GNN_paper import GraphGCN as GNN_GraphGCN
# from models.PG_paper import NodeGCN as PG_NodeGCN
# from models.PG_paper import GraphGCN as PG_GraphGCN

# def string_to_model(paper, dataset):
#     """
#     Given a paper and a dataset return the cooresponding neural model needed for training.
#     :param paper: the paper who's classification model we want to use.
#     :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
#     :returns: torch.nn.module models
#     """
#     if paper == "GNN":
#         if dataset in ['syn1']:
#             return GNN_NodeGCN(10, 4)
#         elif dataset in ['syn2']:
#             return GNN_NodeGCN(10, 8)
#         elif dataset in ['syn3']:
#             return GNN_NodeGCN(10, 2)
#         elif dataset in ['syn4']:
#             return GNN_NodeGCN(10, 2)
#         elif dataset == "ba2":
#             return GNN_GraphGCN(10, 2)
#         elif dataset == "mutag":
#             return GNN_GraphGCN(14, 2)
#         else:
#             raise NotImplementedError
#     elif paper == "PG":
#         if dataset in ['syn1']:
#             return PG_NodeGCN(10, 4)
#         elif dataset in ['syn2']:
#             return PG_NodeGCN(10, 8)
#         elif dataset in ['syn3']:
#             return PG_NodeGCN(10, 2)
#         elif dataset in ['syn4']:
#             return PG_NodeGCN(10, 2)
#         elif dataset == "ba2":
#             return PG_GraphGCN(10, 2)
#         elif dataset == "mutag":
#             return PG_GraphGCN(14, 2)
#         else:
#             raise NotImplementedError
#     else:
#         raise NotImplementedError


# def get_pretrained_path(paper, dataset):
#     """
#     Given a paper and dataset loads the pre-trained model.
#     :param paper: the paper who's classification model we want to use.
#     :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
#     :returns: str; the path to the pre-trined model parameters.
#     """
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     path = f"{dir_path}/pretrained/{paper}/{dataset}/best_model"
#     return path


# def model_selector(paper, dataset, pretrained=True, return_checkpoint=False):
#     """
#     Given a paper and dataset loads accociated model.
#     :param paper: the paper who's classification model we want to use.
#     :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
#     :param pretrained: whter to return a pre-trained model or not.
#     :param return_checkpoint: wheter to return the dict contining the models parameters or not.
#     :returns: torch.nn.module models and optionallly a dict containing it's parameters.
#     """
#     model = string_to_model(paper, dataset)
#     if pretrained:
#         path = get_pretrained_path(paper, dataset)
#         checkpoint = torch.load(path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
#         if return_checkpoint:
#             return model, checkpoint
#     return model

import torch
import os

from models.GNN_paper import NodeGCN as GNN_NodeGCN
from models.GNN_paper import GraphGCN as GNN_GraphGCN
from models.PG_paper import NodeGCN as PG_NodeGCN
from models.PG_paper import GraphGCN as PG_GraphGCN

def string_to_model(paper, dataset):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    if paper == "GNN":
        if dataset in ['syn1']:
            return GNN_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return GNN_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return GNN_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return GNN_NodeGCN(10, 2)
        elif dataset == "ba2":
            return GNN_GraphGCN(10, 2)
        elif dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "PG":
        if dataset in ['syn1']:
            return PG_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return PG_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return PG_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return PG_NodeGCN(10, 2)
        elif dataset == "ba2":
            return PG_GraphGCN(10, 2)
        elif dataset == "mutag":
            return PG_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_pretrained_path(paper, dataset):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: str; the path to the pre-trined model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = f"{dir_path}/pretrained/{paper}/{dataset}/best_model"
    return path


def map_checkpoint_keys(state_dict):
    """
    Map old checkpoint keys to new PyTorch Geometric format.
    
    Key changes between old and new PyTorch Geometric:
    - Old format: conv1.weight with shape [in_features, out_features]
    - New format: conv1.lin.weight with shape [out_features, in_features]
    
    We need to:
    1. Rename keys: conv1.weight -> conv1.lin.weight
    2. Transpose weights: [in_features, out_features] -> [out_features, in_features]
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.weight') and not key.endswith('.lin.weight') and 'conv' in key:
            # Convert conv1.weight -> conv1.lin.weight and transpose
            new_key = key.replace('.weight', '.lin.weight')
            # Transpose from [in_features, out_features] to [out_features, in_features]
            new_state_dict[new_key] = value.t()
        else:
            # Keep other keys as is (bias, lin.weight, etc.)
            new_state_dict[key] = value
    return new_state_dict

def model_selector(paper, dataset, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path(paper, dataset)
        checkpoint = torch.load(path)
        
        # Map old checkpoint keys to new format
        mapped_state_dict = map_checkpoint_keys(checkpoint['model_state_dict'])
        
        try:
            model.load_state_dict(mapped_state_dict)
            print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        except RuntimeError as e:
            print(f"Warning: Could not load pretrained model: {e}")
            
            # Print diagnostic information
            print("\nDiagnostic Info:")
            print("Checkpoint keys and shapes:")
            for key, value in mapped_state_dict.items():
                print(f"  {key}: {value.shape}")
            
            print("\nModel expected keys and shapes:")
            for key, param in model.state_dict().items():
                print(f"  {key}: {param.shape}")
            
            print("\nContinuing with randomly initialized model...")
        
        if return_checkpoint:
            return model, checkpoint
    return model