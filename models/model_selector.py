import torch
import os

from models.GNN_paper import NodeGCN as GNN_NodeGCN
from models.GNN_paper import GraphGCN as GNN_GraphGCN
from models.PG_paper import NodeGCN as PG_NodeGCN
from models.PG_paper import GraphGCN as PG_GraphGCN

def _align_gcn_conv_keys_to_model(model, state_dict):
    """
    Align checkpoint keys for GCNConv layers to match the current torch_geometric version
    used by the instantiated model. Maps between 'conv*.weight' and 'conv*.lin.weight'
    as needed. Leaves other keys untouched.
    """
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # If checkpoint uses new-style '.lin.weight' but model expects old-style '.weight'
        if key.endswith('.lin.weight'):
            alt_key = key.replace('.lin.weight', '.weight')
            if alt_key in model_keys and key not in model_keys:
                new_key = alt_key
        # If checkpoint uses old-style '.weight' but model expects new-style '.lin.weight'
        elif key.endswith('.weight') and 'conv' in key:
            alt_key = key.replace('.weight', '.lin.weight')
            if alt_key in model_keys and key not in model_keys:
                new_key = alt_key
        new_state_dict[new_key] = value
    return new_state_dict

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

def _align_gcn_conv_keys_to_model(model, state_dict):
    model_sd = model.state_dict()
    new_sd = {}
    for k, v in state_dict.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            # If it's a 2D weight and the reversed shape matches, transpose it.
            if v.dim() == 2 and v.shape[::-1] == model_sd[k].shape:
                print(f"[model_selector] Transposing {k}: {tuple(v.shape)} -> {tuple(model_sd[k].shape)}")
                new_sd[k] = v.t().contiguous()
            else:
                # Keep debugging info but skip incompatible tensors
                print(f"[model_selector] Shape mismatch (no simple transpose): {k} {tuple(v.shape)} vs {tuple(model_sd[k].shape)} — skipping.")
                continue
        else:
            new_sd[k] = v
    return new_sd


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
        # Load model with appropriate device mapping
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # Handle compatibility between old and new torch_geometric versions
        state_dict = checkpoint['model_state_dict']
        state_dict = _align_gcn_conv_keys_to_model(model, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[model_selector] missing keys: {missing}")
        print(f"[model_selector] unexpected keys: {unexpected}")

        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model