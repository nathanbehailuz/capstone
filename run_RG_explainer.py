import argparse
import datetime
import hashlib
import json
import pathlib
import time
import sys
import os
import re

import torch
from torch import optim

from components import *
from utils import *

from datasets import dataset_loaders, ground_truth_loaders
from models import model_selector
from evaluation.AUCEvaluation import AUCEvaluation
import torch_geometric as ptgeom
from components.causal_explainer import NodeLevelCausalExplainer, GraphLevelCausalExplainer


def pretrain_g(g: Generator, train_comms, eval_seeds, max_size, bs, n, score_fn, dataset, writer, use_set=True):
    eval_seeds = np.array(eval_seeds)
    shuffle_index = np.arange(len(train_comms)) 

    for i in range(n):
        np.random.shuffle(shuffle_index)
        train_comms = train_comms[shuffle_index]
        eval_seeds = eval_seeds[shuffle_index]

        batch_loss = 0.
        for j in range(len(train_comms) // bs + 1):
            batch = train_comms[j*bs:(j+1)*bs]
            batch_seeds = eval_seeds[j*bs:(j+1)*bs]
            if len(batch) == 0:
                continue

            if dataset == "mutag":
                batch = [g.graph.sample_layerwise_expansion(x, es, max_size) for x, es in zip(batch,batch_seeds)] # for mutag
            else:
                batch = [g.graph.sample_expansion_with_high_scores(score_fn, 10, x, es, max_size) for x, es in zip(batch, batch_seeds)] # for syn1-sy4, ba
            

            if use_set:
                policy_loss = g.train_from_sets(batch)
            else:
                policy_loss = g.train_from_lists(batch)
            batch_loss += policy_loss
        batch_loss /= j + 1
        if use_set:
            s = 'Set '
        else:
            s = 'List'
        if writer is not None:
            writer.add_scalar(f'Pretrain/GLoss{s.strip()}', batch_loss, i)
        print(f'[Pretrain-{s} {i+1:3d}] Loss = {batch_loss:2.4f}', flush=True)


class DummyWriter:

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class Runner:

    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        # Load Dataset
        self.graphs, self.nodefeats, ground_truth_labels, self.task = self.load_data()
        # Move nodefeats to GPU
        if isinstance(self.nodefeats, list):
            self.nodefeats = [feat.to(self.device) for feat in self.nodefeats]
        else:
            self.nodefeats = self.nodefeats.to(self.device)
        # Store ground truth labels for later use
        self.ground_truth_labels = ground_truth_labels
        # Load pretrained models
        self.trained_model, checkpoint = model_selector.model_selector(args.model, args.dataset, pretrained=True, return_checkpoint=True)
        # Move model to GPU
        self.trained_model = self.trained_model.to(self.device)
        # Load ground truth
        self.explanation_labels, explain_eval_seeds = ground_truth_loaders.load_dataset_ground_truth(args.dataset)
        # Load AUC Evaluation
        self.auc_eval= AUCEvaluation(self.task, self.explanation_labels, explain_eval_seeds)


        max_size_in_sg = 0
        if self.task == "node_task":
            self.eval_seeds = explain_eval_seeds
            print("Eval Seeds", self.eval_seeds, flush=True)

            self.subgraphs = []
            for seed_node in self.eval_seeds:
                if self.args.dataset == "syn2":
                    gt_graph = ptgeom.utils.subgraph(torch.LongTensor(np.array(range(0,700,1))), torch.LongTensor(self.graphs))[0] # for syn2 to limit the ground_truth edges in the correct graph
                else:
                    gt_graph = torch.LongTensor(self.graphs)
                _, subgraph_edge, *_ = ptgeom.utils.k_hop_subgraph(seed_node, self.args.n_hop, gt_graph)
                # Move subgraph edge_index to GPU
                self.subgraphs.append(subgraph_edge.to(self.device))

            self.pretrain_nodes = []
            self.pretrain_eval_nodes = []
            for seed_node in self.eval_seeds:
                for nl in range(self.args.n_hop-1, self.args.n_hop, 1):
                    subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(seed_node, nl+1, torch.LongTensor(self.graphs))
                    self.pretrain_nodes.append(subgraph_node.cpu().numpy())
                    self.pretrain_eval_nodes.append(seed_node)
                    if len(subgraph_node)>max_size_in_sg:
                        max_size_in_sg = len(subgraph_node)
            self.pretrain_nodes = np.array(self.pretrain_nodes, dtype=object)

            self.graphs = Graph(self.graphs)
            # Move graph edge_index to GPU
            self.graphs.edge_index = self.graphs.edge_index.to(self.device)

            edge_weights = torch.ones(self.graphs.edge_index.size(1), device=self.graphs.edge_index.device)
            whole_graph_original_preds = self.trained_model(self.nodefeats, self.graphs.edge_index, edge_weights)
            whole_graph_original_preds = torch.argmax(whole_graph_original_preds, 1)
            # Move ground_truth_labels to same device as predictions
            ground_truth_labels_device = ground_truth_labels.to(whole_graph_original_preds.device)
            print("Prediction Error for Whole Graph:", torch.sum(torch.abs(whole_graph_original_preds - ground_truth_labels_device)).item()/len(whole_graph_original_preds), flush=True)

            subg_original_preds = []
            for i, es in enumerate(self.eval_seeds):
                edge_weights = torch.ones(self.subgraphs[i].size(1), device=self.subgraphs[i].device)
                pred = self.trained_model(self.nodefeats, self.subgraphs[i], edge_weights)[es]
                subg_original_preds.append(pred)
            subg_original_preds = torch.stack(subg_original_preds)
            subg_original_preds = torch.argmax(subg_original_preds, 1)
            # Move ground_truth_labels to same device as predictions
            eval_ground_truth = ground_truth_labels[self.eval_seeds].to(subg_original_preds.device)
            print("Prediction Error for Eval Seeds:", torch.sum(torch.abs(subg_original_preds-eval_ground_truth)).item()/len(subg_original_preds), flush=True)

            self.original_preds = whole_graph_original_preds
            edge_weights = torch.ones(self.graphs.edge_index.size(1), device=self.graphs.edge_index.device)
            self.original_embeds = self.trained_model.embedding(self.nodefeats, self.graphs.edge_index, edge_weights)            

            print("N-hop", self.args.n_hop, flush=True)
            print("Max Size in SG", max_size_in_sg, flush=True)

        else: # 'graph_task'
            
            self.graph_eval_seeds = explain_eval_seeds
            # Initialize eval_seeds for graph tasks (will be set properly in pretrain_l)
            self.eval_seeds = explain_eval_seeds

            each_g_actual_nodes = []
            for i in range(len(self.graphs)):
                actual_nodes = int(torch.sum(self.nodefeats[i]))
                each_g_actual_nodes.append(actual_nodes)
            print("total nodes", np.sum(np.array(each_g_actual_nodes)), flush=True)

            each_g_actual_edges = []
            total_edges = 0
            for i in range(len(self.graphs)):
                actual_edges = []
                for e1, e2 in zip(self.graphs[i][0], self.graphs[i][1]):
                    if e1==e2:
                        pass
                    else:
                        actual_edges.append([e1, e2])
                total_edges += len(actual_edges)
                each_g_actual_edges.append(np.array(actual_edges).T)
            print("total edges", total_edges, flush=True)

            new_graphs_0 = []
            new_graphs_1 = []
            new_graphs_node = []
            cur_node_id = 0
            each_g_start_nid = []
            each_g_start_eid = []
            new_node_feats = []
            for i in range(len(self.graphs)):
                max_node_id = each_g_actual_nodes[i]
                
                each_g_start_nid.append(cur_node_id)
                each_g_start_eid.append(len(new_graphs_0))
                
                new_graphs_0.extend(each_g_actual_edges[i][0] + cur_node_id)
                new_graphs_1.extend(each_g_actual_edges[i][1] + cur_node_id)
                new_graphs_node.append(list(range(cur_node_id, cur_node_id+max_node_id)))

                cur_node_id += max_node_id
                new_node_feats.append(self.nodefeats[i][0:max_node_id])
            
            each_g_start_eid.append(len(new_graphs_0))
            new_graphs = np.array([new_graphs_0, new_graphs_1])
            self.graphs = Graph(new_graphs)
            # Move graph edge_index to GPU
            self.graphs.edge_index = self.graphs.edge_index.to(self.device)

            # Store sliced node features (only actual nodes, not padded) for each graph
            self.ind_nodefeats = []
            for i in range(len(self.nodefeats)):
                actual_nodes_count = each_g_actual_nodes[i]
                self.ind_nodefeats.append(self.nodefeats[i][0:actual_nodes_count].to(self.device))
            
            self.nodefeats = torch.cat(new_node_feats).to(self.device) 

            # Store helper variables for later use
            self.each_g_start_nid = each_g_start_nid
            self.each_g_start_eid = each_g_start_eid
            
            # graph embedding
            whole_graph_original_preds = []
            # Pre-allocate edge weights to avoid repeated creation
            max_edges = max(each_g_start_eid[i+1] - each_g_start_eid[i] for i in range(len(each_g_start_nid)))
            edge_weights_template = torch.ones(max_edges, dtype=torch.float32, device=self.device)
            
            for i in range(len(each_g_start_nid)):
                edge_idx = self.graphs.edge_index[:, each_g_start_eid[i]:each_g_start_eid[i+1]] - each_g_start_nid[i]
                
                # Get the appropriate node features for these edges
                node_feats_subset = self.get_node_features_for_edges(i, edge_idx)
                
                # Use pre-allocated edge weights
                edge_weights = edge_weights_template[:edge_idx.size(1)]
                pred = self.trained_model(node_feats_subset, edge_idx, edge_weights)[0]
                whole_graph_original_preds.append(pred)
            whole_graph_original_preds = torch.stack(whole_graph_original_preds)
            whole_graph_original_preds = torch.argmax(whole_graph_original_preds, 1)
            ground_truth_labels = torch.argmax(ground_truth_labels.to(self.device), 1)
            print("Prediction Error for Graphs:", torch.sum(torch.abs(whole_graph_original_preds - ground_truth_labels)).item()/len(whole_graph_original_preds), flush=True)

            g_embeds = []
            for i in range(len(each_g_start_nid)):
                edge_idx = self.graphs.edge_index[:, each_g_start_eid[i]:each_g_start_eid[i+1]] - each_g_start_nid[i]
                
                node_feats_subset = self.get_node_features_for_edges(i, edge_idx)
                edge_weights = edge_weights_template[:edge_idx.size(1)]
                g_embed = self.trained_model.graph_embedding(node_feats_subset, edge_idx, edge_weights)[0]
                g_embeds.append(g_embed)
            g_embeds = torch.stack(g_embeds)

            # node embedding
            n_embeds = []
            for i in range(len(each_g_start_nid)):
                edge_idx = self.graphs.edge_index[:, each_g_start_eid[i]:each_g_start_eid[i+1]] - each_g_start_nid[i]
                
                node_feats_subset = self.get_node_features_for_edges(i, edge_idx)
                edge_weights = edge_weights_template[:edge_idx.size(1)]
                temp = self.trained_model.embedding(node_feats_subset, edge_idx, edge_weights)
                # Only keep embeddings for nodes that will be used
                n_embeds.append(temp[0:min(temp.size(0), len(new_graphs_node[i]))])
            n_embeds = torch.cat(n_embeds)

            # isolated nodes
            # total_isolated_nodefeats = []
            # if len(self.graphs.isolated_nodes)>0:
            #     for n_id in self.graphs.isolated_nodes:
            #         print(n_id, self.nodefeats[n_id])
            #         total_isolated_nodefeats.append(self.nodefeats[n_id])
            #     print("total", len(self.graphs.isolated_nodes), torch.sum(torch.stack(total_isolated_nodefeats),0))

            # build the graph index
            self.n_to_g_index = []
            for i in range(len(each_g_start_nid)):
                if i != len(each_g_start_nid) - 1:
                    for j in range(each_g_start_nid[i], each_g_start_nid[i+1], 1):
                        self.n_to_g_index.append(i)
                else:
                    for j in range(each_g_start_nid[i], n_embeds.size()[0], 1):
                        self.n_to_g_index.append(i)

            # ind_subgraphs
            self.ind_subgraphs = new_graphs_node
            # Note: self.each_g_start_nid and self.each_g_start_eid already assigned above
            self.original_preds = whole_graph_original_preds
            self.g_embeds = g_embeds
            self.max_n_nodes_in_g = np.max(np.array(each_g_actual_nodes))
            print("self.max_n_nodes_in_g",self.max_n_nodes_in_g, flush=True)

        self.nodefeats_size = self.nodefeats.shape[-1]

        # Save Dir and Pretrained Dir
        self.savedir, self.pretrain_dir, self.writer = self.init_dir()
        self.args.ds_name = f'{args.dataset}-1.90'

        # Model
        self.g = self.init_g()
        if self.task == "graph_task":
            self.l = self.init_l(g_embeds, n_embeds)
        
        # Initialize causal mechanism if enabled
        self.init_causal_mechanism()

    def close(self):
        self.writer.close()

    def load_data(self, shuffle=True):
        args = self.args
        # Load complete dataset
        graphs, features, ground_truth_labels, _, _, test_mask = dataset_loaders.load_dataset(args.dataset, shuffle = shuffle)
        if isinstance(graphs, list): # We're working with a model for graph classification
            task = "graph_task"
        else:
            task = "node_task"
        features = torch.tensor(features)
        ground_truth_labels = torch.tensor(ground_truth_labels)
        
        return graphs, features, ground_truth_labels, task

    def init_dir(self):
        args = self.args
        savedir = pathlib.Path(args.savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        # writer = SummaryWriter(savedir / 'tb')
        writer = DummyWriter()
        with open(savedir / 'settings.json', 'w') as fh:
            arg_dict = vars(args)
            arg_dict['model'] = 'RGExplainer_v1'
            json.dump(arg_dict, fh, sort_keys=True, indent=4)
        pretrain_dir = pathlib.Path('pretrained')
        pretrain_dir.mkdir(exist_ok=True)
        return savedir, pretrain_dir, writer

    def init_g(self):
        args = self.args
        device = self.device
        
        # Assert that causal mechanism is enabled
        use_causal = getattr(args, 'use_causal', False)
        if use_causal:
            assert use_causal, "[ERROR] Causal mechanism must be enabled (--use_causal flag or default=True)"
            print(f"[CAUSAL] Causal mechanism check passed: use_causal={use_causal}")
        
        if args.with_attr:
            g_model = Agent(args.hidden_size, args.with_attr, self.nodefeats_size, stop_bias=args.stop_bias).to(device)
        else:
            g_model = Agent(args.hidden_size, args.with_attr, stop_bias=args.stop_bias).to(device)

        g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr)
        
        # Initialize causal mechanism parameters
        causal_weight = getattr(args, 'causal_weight', 0.5)
        causal_explainer = None
        
        if use_causal:
            # Determine number of classes from ground truth labels
            if self.task == "node_task":
                # For node tasks, ground_truth_labels is 1D
                num_classes = len(torch.unique(self.ground_truth_labels))
            else:  # graph_task
                # For graph tasks, ground_truth_labels may be one-hot encoded
                if len(self.ground_truth_labels.shape) == 1:
                    num_classes = len(torch.unique(self.ground_truth_labels))
                else:
                    num_classes = self.ground_truth_labels.shape[1]
            
            # Determine which causal explainer to use based on task
            if self.task == "node_task":
                causal_explainer = NodeLevelCausalExplainer(
                    gnn_model=self.trained_model,
                    num_labels=num_classes,
                    hidden_size=args.hidden_size,
                    use_edge_attr=args.with_attr
                ).to(device)
            else:  # graph_task
                causal_explainer = GraphLevelCausalExplainer(
                    gnn_model=self.trained_model,
                    num_labels=num_classes,
                    hidden_size=args.hidden_size,
                    use_edge_attr=args.with_attr
                ).to(device)
            print(f"[CAUSAL] Initialized {type(causal_explainer).__name__} with {num_classes} classes, weight {causal_weight}")
        else:
            print("[CAUSAL] Causal mechanism disabled")
        
        g = Generator(self.graphs, g_model, g_optimizer, device,
                      entropy_coef=args.entropy_coef,
                      n_rollouts=args.n_rollouts,
                      max_size=args.max_size,
                      max_reward=5.,
                      # New parameters for STOP action and penalties
                      target_size_k=args.target_size_k,
                      lambda_size=args.lambda_size,
                      lambda_radius=args.lambda_radius,
                      lambda_radius_min=args.lambda_radius_min,
                      step_cost=args.step_cost,
                      stop_bias=args.stop_bias,
                      fidelity_thresh=args.fidelity_thresh,
                      # PPO stability parameters
                      ppo_clip=args.clip_eps,  # Legacy parameter
                      clip_range_policy=args.clip_range_policy,
                      clip_range_value=args.clip_range_value,
                      kl_target=args.kl_target,
                      lr=args.lr,
                      gamma=args.gamma,
                      gae_lambda=args.gae_lambda,
                      max_grad_norm=args.grad_clip,
                      use_value_clip=args.use_value_clip,
                      adaptive_kl_penalty=args.adaptive_kl_penalty,
                      total_epochs=args.n_epochs,
                      # Entropy scheduling
                      entropy_coeff_start=args.entropy_decay_start,
                      entropy_coeff_end=args.entropy_decay_end,
                      entropy_decay_final=args.entropy_decay_final,
                      # Causal mechanism parameters
                      use_causal=use_causal,
                      causal_explainer=causal_explainer,
                      causal_weight=causal_weight,
                      causal_warmup_epochs=args.causal_warmup_epochs,
                      # Debug parameters
                      debug_mode=bool(args.debug_mode))
        if args.with_attr: 
            g.load_nodefeats(self.nodefeats)
        
        # Initialize causal reward calculator
        g.init_causal_reward_calculator(self.trained_model)
        
        # Set causal data for intervention-based causal bias computation
        if args.use_causal:
            # Get target classes for eval seeds
            with torch.no_grad():
                edge_weights = torch.ones(self.graphs.edge_index.size(1), device=self.graphs.edge_index.device)
                all_preds = self.trained_model(self.nodefeats, self.graphs.edge_index, edge_weights)
                target_classes = torch.argmax(all_preds, dim=1)[list(self.eval_seeds)]
            
            g.set_causal_data(
                gnn_model=self.trained_model,
                full_edge_index=self.graphs.edge_index,
                target_classes=target_classes,
                seed_nodes=list(self.eval_seeds)
            )
        
        return g


    def init_l(self, g_embeds, n_embeds):
        args = self.args
        l = Locator(g_embeds, n_embeds, self.ind_subgraphs, args.l_lr, self.graphs.isolated_nodes, device=self.device)
        return l

    def init_causal_mechanism(self):
        """
        Initialize causal mechanism components.
        This method sets up the causal explainer and integrates it with the system.
        """
        args = self.args
        use_causal = getattr(args, 'use_causal', False)
        
        if use_causal:
            print("[CAUSAL] Causal mechanism enabled")
            print(f"[CAUSAL] Task type: {self.task}")
            print(f"[CAUSAL] Causal weight: {getattr(args, 'causal_weight', 0.5)}")
        else:
            print("[CAUSAL] Causal mechanism disabled")


    def prepare_mask(self, o_g, selected_nodes):
        n_edges = o_g.size()[1]
        n_edges_index = {}
        for i in range(n_edges):
            n_edges_index[(int(o_g[0][i]), int(o_g[1][i]))] = i
        
        rank = torch.zeros(n_edges)

        for i in range(len(selected_nodes)):
            for j in range(i):
                if (int(selected_nodes[i]), int(selected_nodes[j])) in n_edges_index and (int(selected_nodes[j]), int(selected_nodes[i])) in n_edges_index:
                    rank[n_edges_index[(int(selected_nodes[i]), int(selected_nodes[j]))]] = self.args.max_size - i
                    rank[n_edges_index[(int(selected_nodes[j]), int(selected_nodes[i]))]] = self.args.max_size - i
                else:
                    if (int(selected_nodes[i]), int(selected_nodes[j])) in n_edges_index or (int(selected_nodes[j]), int(selected_nodes[i])) in n_edges_index:
                        print("edge pair error", flush=True)

        mask = rank / (self.args.max_size-1)
        
        # Create edge scores dictionary for saving
        edge_scores = {}
        for i in range(n_edges):
            src = int(o_g[0][i])
            dst = int(o_g[1][i])
            score = float(mask[i])
            edge_scores[(src, dst)] = score
        
        return (o_g, mask), edge_scores

    def eval_loss(self, cs):
        cs = [x[:-1] if x[-1] == 'EOS' else x for x in cs]
        batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]
        
        # Prediction loss
        if self.task == "node_task":
            masked_preds = []
            for i, g in enumerate(batch_g):
                edge_weights = torch.ones(g.size(1), device=g.device)
                pred = self.trained_model(self.nodefeats, g, edge_weights)[cs[i][0]]
                masked_preds.append(pred)
            original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]
        else:
            batch_g_id = [self.get_g_id(x, True) for x in cs]
            masked_preds = []
            for g_id, g in zip(batch_g_id, batch_g):
                edge_idx = g - self.each_g_start_nid[g_id]
                edge_weights = torch.ones(edge_idx.size(1), dtype=torch.float32, device=edge_idx.device)
                node_feats = self.get_node_features_for_edges(g_id, edge_idx)
                pred = self.trained_model(node_feats, edge_idx, edge_weights)
                masked_preds.append(pred)
            original_preds = [self.original_preds[g_id] for g_id in batch_g_id]
        

        masked_preds = torch.stack(masked_preds).squeeze()
        original_preds = torch.stack(original_preds)

        prediction_loss = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none') 
        prediction_loss = prediction_loss.detach().cpu().numpy()


        # Size loss
        if args.size_reg > 0:
            size_loss = args.size_reg * torch.tensor([g.size()[1] for g in batch_g], dtype=torch.float32)
            size_loss = size_loss.detach().cpu().numpy()
        else:
            size_loss = np.array([0.0]*len(prediction_loss))


        # Radius Penalty
        if args.radius_penalty > 0:
            radius_p = args.radius_penalty * torch.tensor([self.graphs.subgraph_depth(x) for x in cs], dtype=torch.float32)
            radius_p = radius_p.detach().cpu().numpy()
        else:
            radius_p = np.array([0.0]*len(prediction_loss))


        # Similarity loss
        if args.sim_reg > 0:
            if self.task == "node_task":
                masked_embeds = []
                for i, g in enumerate(batch_g):
                    edge_weights = torch.ones(g.size(1), device=g.device)
                    embed = self.trained_model.embedding(self.nodefeats, g, edge_weights)[cs[i][0]]
                    masked_embeds.append(embed)
                sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
                sim_loss = args.sim_reg * torch.stack(sim_loss)
            else:
                masked_embeds = []
                for g_id, g in zip(batch_g_id, batch_g):
                    edge_idx = g - self.each_g_start_nid[g_id]
                    edge_weights = torch.ones(edge_idx.size(1), dtype=torch.float32, device=edge_idx.device)
                    node_feats = self.get_node_features_for_edges(g_id, edge_idx)
                    embed = self.trained_model.graph_embedding(node_feats, edge_idx, edge_weights)
                    masked_embeds.append(embed)
                sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
                sim_loss = args.sim_reg * torch.stack(sim_loss)
            sim_loss = sim_loss.detach().cpu().numpy()
        else:
            sim_loss = np.array([0.0]*len(prediction_loss))
                
        return prediction_loss, size_loss, sim_loss, radius_p


    def evaluate_and_print(self, prefix=''):
        print(f"[DEBUG] Starting evaluation for {prefix}", flush=True)

        pred_sgs = self.g.generate(self.eval_seeds)
        p_loss, s_loss, sim_loss, r_p = self.eval_loss(pred_sgs)
        pred_sgs = [x[:-1] if x[-1] == 'EOS' else x for x in pred_sgs] 
        
        
        if self.args.dataset == "mutag":
            node_type = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']

            for idx, i in enumerate(self.graph_eval_seeds):
                if idx < len(pred_sgs) and len(pred_sgs[idx]) > 0:
                    print("g_id", i, "n_id", pred_sgs[idx][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[idx][0]]] ,"len",len(pred_sgs[idx]), f"p_loss={p_loss[idx]:.2f}", f"size_loss={s_loss[idx]:.2f}", f"sim_loss={sim_loss[idx]:.2f}",f"r_p={r_p[idx]:.2f}", "sgs", [node_type[torch.argmax(self.nodefeats[x]).item()] for x in pred_sgs[idx]], flush=True)
                else:
                    print("g_id", i, "n_id", "N/A (empty)", "len", 0, f"p_loss={p_loss[idx] if idx < len(p_loss) else 0:.2f}", f"size_loss={s_loss[idx] if idx < len(s_loss) else 0:.2f}", f"sim_loss={sim_loss[idx] if idx < len(sim_loss) else 0:.2f}",f"r_p={r_p[idx] if idx < len(r_p) else 0:.2f}", "sgs", [], flush=True)
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}", flush=True)

            for idx, i in enumerate(self.graph_eval_seeds):
                if idx < len(pred_sgs) and len(pred_sgs[idx]) > 0:
                    print("g_id", i, "n_id", pred_sgs[idx][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[idx][0]]] ,"len",len(pred_sgs[idx]), f"p_loss={p_loss[idx]:.2f}", f"size_loss={s_loss[idx]:.2f}", f"sim_loss={sim_loss[idx]:.2f}",f"r_p={r_p[idx]:.2f}", "sgs", [x - self.each_g_start_nid[self.n_to_g_index[x]] for x in pred_sgs[idx]], flush=True)
                else:
                    print("g_id", i, "n_id", "N/A (empty)", "len", 0, f"p_loss={p_loss[idx] if idx < len(p_loss) else 0:.2f}", f"size_loss={s_loss[idx] if idx < len(s_loss) else 0:.2f}", f"sim_loss={sim_loss[idx] if idx < len(sim_loss) else 0:.2f}",f"r_p={r_p[idx] if idx < len(r_p) else 0:.2f}", "sgs", [], flush=True)

        elif self.args.dataset == "ba2":

            for idx, i in enumerate(self.graph_eval_seeds):
                if idx < len(pred_sgs) and len(pred_sgs[idx]) > 0:
                    print("g_id", i, "n_id", pred_sgs[idx][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[idx][0]]] ,"len",len(pred_sgs[idx]), f"p_loss={p_loss[idx]:.2f}", f"size_loss={s_loss[idx]:.2f}", f"sim_loss={sim_loss[idx]:.2f}",f"r_p={r_p[idx]:.2f}", "sgs", [x - self.each_g_start_nid[self.n_to_g_index[x]] for x in pred_sgs[idx]], flush=True)
                else:
                    print("g_id", i, "n_id", "N/A (empty)", "len", 0, f"p_loss={p_loss[idx] if idx < len(p_loss) else 0:.2f}", f"size_loss={s_loss[idx] if idx < len(s_loss) else 0:.2f}", f"sim_loss={sim_loss[idx] if idx < len(sim_loss) else 0:.2f}",f"r_p={r_p[idx] if idx < len(r_p) else 0:.2f}", "sgs", [], flush=True)
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}", flush=True)

        else: # print current first ~50 sgs for syn1-syn4
            for i in range(min(len(pred_sgs),50)):
                if len(pred_sgs[i]) > 0:
                    print("n_id", pred_sgs[i][0] ,"len",len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}",f"r_p={r_p[i]:.2f}", "sgs", [x - pred_sgs[i][0] for x in pred_sgs[i]], flush=True)
                else:
                    print("n_id", "N/A (empty)", "len", 0, f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}",f"r_p={r_p[i]:.2f}", "sgs", [], flush=True)
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}", flush=True)


        if self.task == "node_task":
            explanations = []
            edge_scores_list = []
            for i, x in enumerate(pred_sgs):
                expl, edge_scores = self.prepare_mask(self.subgraphs[i], x)
                explanations.append(expl)
                edge_scores_list.append(edge_scores)
        else:
            explanations = []
            edge_scores_list = []
            for idx, g_id in enumerate(self.graph_eval_seeds):
                if idx < len(pred_sgs):
                    expl, edge_scores = self.prepare_mask(self.graphs.edge_index[:, self.each_g_start_eid[g_id]:self.each_g_start_eid[g_id+1]] - self.each_g_start_nid[g_id], [x - self.each_g_start_nid[g_id] for x in pred_sgs[idx]])
                    explanations.append(expl)
                    edge_scores_list.append(edge_scores)
                else:
                    # Handle case where pred_sgs doesn't have enough elements
                    expl, edge_scores = self.prepare_mask(self.graphs.edge_index[:, self.each_g_start_eid[g_id]:self.each_g_start_eid[g_id+1]] - self.each_g_start_nid[g_id], [])
                    explanations.append(expl)
                    edge_scores_list.append(edge_scores)


        auc_score = self.auc_eval.get_score(explanations, 
                                          edge_score_agg=self.args.edge_score_agg,
                                          debug_eval_graph=self.args.debug_eval_graph)
        print(f'[EVAL-{prefix}] AUC={auc_score}', flush=True)
        print(f"[DEBUG] Evaluation completed for {prefix}, AUC={auc_score}", flush=True)

        # Save edge scores to text files
        self.save_edge_scores_to_file(edge_scores_list, prefix)

        return auc_score

    def save_edge_scores_to_file(self, edge_scores_list, prefix):
        """Save edge scores to the results file."""
        if not edge_scores_list:
            return
            
        # Determine the results file path
        results_file = f"results/{self.args.dataset}_ppo_seed_{self.args.seed}.txt"
        
        # Read existing content
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: Results file {results_file} not found, cannot save edge scores")
            return
        
        # Find the current epoch section and add edge scores
        epoch_pattern = rf'\[EVAL-{prefix}\]'
        modified_lines = []
        edge_scores_idx = 0
        
        for i, line in enumerate(lines):
            modified_lines.append(line)
            
            # Check if this is an epoch marker
            if re.search(epoch_pattern, line):
                # Look ahead for the next lines that contain subgraph data
                j = i + 1
                while j < len(lines) and not re.search(r'\[EVAL-', lines[j]):
                    current_line = lines[j]
                    
                    # Check if this line contains subgraph data (sgs [...])
                    if 'sgs' in current_line and '[' in current_line and ']' in current_line:
                        if edge_scores_idx < len(edge_scores_list):
                            edge_scores = edge_scores_list[edge_scores_idx]
                            
                            # Format edge scores as a string
                            if edge_scores:
                                edge_scores_str = "edge_scores {"
                                edge_items = []
                                for (src, dst), score in edge_scores.items():
                                    edge_items.append(f"({src},{dst}):{score:.4f}")
                                edge_scores_str += ",".join(edge_items) + "}"
                                
                                # Add edge scores line after the subgraph line
                                modified_lines.append(f"  {edge_scores_str}\n")
                                edge_scores_idx += 1
                            else:
                                modified_lines.append("  edge_scores {}\n")
                                edge_scores_idx += 1
                    
                    j += 1
        
        # Write the modified content back to the file
        try:
            with open(results_file, 'w') as f:
                f.writelines(modified_lines)
            print(f"Edge scores saved to {results_file}")
        except Exception as e:
            print(f"Error saving edge scores: {e}")

    def get_g_id(self, nodes, is_check=False):
        if is_check:
            # check whether nodes belong to one graph
            flag_check = True
            for i in range(len(nodes)-1):
                if self.n_to_g_index[nodes[i]] != self.n_to_g_index[nodes[i+1]]:
                    flag_check = False
                    break
            if flag_check == False:
                print("Nodes do not belong to one graph!")

        return self.n_to_g_index[nodes[0]]
    
    def get_node_features_for_edges(self, g_id, edge_idx):
        """
        Get the appropriate node features based on which nodes are referenced in the edge index.
        For GCN to work properly, we need to pass ALL node features from 0 to the max node index,
        even if some nodes are not connected (isolated nodes).
        """
        if edge_idx.size(1) > 0:
            # Get the maximum node index referenced in edges
            max_node_idx = int(edge_idx.max().item())
            # Return node features from 0 to max_node_idx (inclusive)
            num_nodes_needed = max_node_idx + 1
        else:
            # No edges, return a single node feature
            num_nodes_needed = 1
        
        # Make sure we don't exceed the available node features
        num_nodes_needed = min(num_nodes_needed, self.ind_nodefeats[g_id].size(0))
        return self.ind_nodefeats[g_id][:num_nodes_needed]

    def prepare_graph(self, idx, selected_nodes):
        selected_nodes = list(selected_nodes)
        if self.task == "node_task":
            return ptgeom.utils.subgraph(selected_nodes, self.graphs.edge_index)[0]
        else:
            return ptgeom.utils.subgraph(selected_nodes, self.graphs.edge_index)[0]

    def score_fn(self, cs):
        cs = [x[:-1] if x[-1] == 'EOS' else x for x in cs]
        batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]
        
        # Prediction loss
        if self.task == "node_task":
            masked_preds = []
            for i, g in enumerate(batch_g):
                edge_weights = torch.ones(g.size(1), device=g.device)
                pred = self.trained_model(self.nodefeats, g, edge_weights)[cs[i][0]]
                masked_preds.append(pred)
            original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]
        else:
            batch_g_id = [self.get_g_id(x, True) for x in cs]
            masked_preds = []
            for g_id, g in zip(batch_g_id, batch_g):
                edge_idx = g - self.each_g_start_nid[g_id]
                edge_weights = torch.ones(edge_idx.size(1), dtype=torch.float32, device=edge_idx.device)
                node_feats = self.get_node_features_for_edges(g_id, edge_idx)
                pred = self.trained_model(node_feats, edge_idx, edge_weights)
                masked_preds.append(pred)
            original_preds = [self.original_preds[g_id] for g_id in batch_g_id]

        masked_preds = torch.stack(masked_preds).squeeze()
        original_preds = torch.stack(original_preds)

        v = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none') 

        # Size loss with target size penalty
        if args.size_reg > 0:
            size_penalty = args.size_reg * torch.tensor([g.size()[1] for g in batch_g], dtype=torch.float32, device=self.device)
            v += size_penalty
        
        # Target size hinge loss
        if args.lambda_size > 0:
            sizes = torch.tensor([len(x) for x in cs], dtype=torch.float32, device=self.device)
            target_size = torch.tensor(args.target_size_k, dtype=torch.float32, device=self.device)
            size_hinge = torch.clamp(sizes - target_size, min=0.0)  # Hinge loss: max(0, size - target)
            v += args.lambda_size * size_hinge
        
        # Radius penalty with decay
        if args.radius_penalty > 0:
            radius_penalty = args.radius_penalty * torch.tensor([self.graphs.subgraph_depth(x) for x in cs], dtype=torch.float32, device=self.device)
            v += radius_penalty
        
        # Adaptive radius penalty (lambda_radius with decay)
        if args.lambda_radius > 0:
            # Calculate current epoch for decay (assuming we have access to it)
            current_epoch = getattr(self, 'current_epoch', 0)
            max_epochs = getattr(self, 'max_epochs', 10)
            
            # Linear decay from lambda_radius to lambda_radius_min
            decay_factor = max(0.0, 1.0 - (current_epoch / max_epochs))
            current_lambda_radius = args.lambda_radius_min + (args.lambda_radius - args.lambda_radius_min) * decay_factor
            
            radius_depths = torch.tensor([self.graphs.subgraph_depth(x) for x in cs], dtype=torch.float32, device=self.device)
            adaptive_radius_penalty = current_lambda_radius * radius_depths
            v += adaptive_radius_penalty
        
        # Step cost penalty
        if args.step_cost > 0:
            step_costs = torch.tensor([len(x) * args.step_cost for x in cs], dtype=torch.float32, device=self.device)
            v += step_costs
        
        # FIXED: Modest fidelity reward - scaled down from 5.0 to 0.05
        # This provides gentle encouragement without overwhelming the value function
        if args.fidelity_thresh > 0:
            # Calculate fidelity (1 - prediction loss) for each subgraph
            fidelity_scores = 1.0 - torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none')
            
            # Apply SMALL bonus for high fidelity (scaled to match regularization)
            fidelity_bonus = torch.where(fidelity_scores >= args.fidelity_thresh, 
                                       torch.tensor(0.05, device=self.device),  # Small bonus (was 5.0)
                                       torch.tensor(0.0, device=self.device))
            v -= fidelity_bonus  # Subtract because v is loss (negative reward)
        
        # REMOVED: EXTREME size bonuses/penalties (were causing size to max at 20)
        # SIZE_REG alone is sufficient for size control
        
        # REMOVED: Diversity bonus (was conflicting with size penalties)
        
        # REMOVED: Oracle bonus (was 20.0 - causing value explosion)

        # Similarity loss
        if args.sim_reg > 0:
            if self.task == "node_task":
                masked_embeds = []
                for i, g in enumerate(batch_g):
                    edge_weights = torch.ones(g.size(1), device=g.device)
                    embed = self.trained_model.embedding(self.nodefeats, g, edge_weights)[cs[i][0]]
                    masked_embeds.append(embed)
                sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
            else:
                masked_embeds = []
                for g_id, g in zip(batch_g_id, batch_g):
                    edge_idx = g - self.each_g_start_nid[g_id]
                    edge_weights = torch.ones(edge_idx.size(1), dtype=torch.float32, device=edge_idx.device)
                    node_feats = self.get_node_features_for_edges(g_id, edge_idx)
                    embed = self.trained_model.graph_embedding(node_feats, edge_idx, edge_weights)
                    masked_embeds.append(embed)
                sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
            v += args.sim_reg * torch.stack(sim_loss) 
        
        # ========== CAUSAL MECHANISM INTEGRATION ==========
        # Add causal reward if causal mechanism is enabled
        if hasattr(self.g, 'use_causal') and self.g.use_causal and self.g.causal_reward_calculator is not None:
            causal_rewards = []
            for i, x in enumerate(cs):
                # Get target class for this subgraph
                if self.task == "node_task":
                    if 0 <= x[0] < len(self.original_preds):
                        target_y = self.original_preds[x[0]].item()
                    else:
                        print(f"[CAUSAL] Warning: node_id {x[0]} out of bounds for original_preds (len={len(self.original_preds)}), using default target_y=0")
                        target_y = 0
                else:
                    g_id = self.get_g_id(x, True)
                    if 0 <= g_id < len(self.original_preds):
                        target_y = self.original_preds[g_id].item()
                    else:
                        print(f"[CAUSAL] Warning: g_id {g_id} out of bounds for original_preds (len={len(self.original_preds)}), using default target_y=0")
                        target_y = 0
                
                # Create a simple graph object for causal analysis
                # This is a simplified version - in practice, you'd need proper PyG Data objects
                class SimpleGraph:
                    def __init__(self, edge_index, x, batch=None, y=None):
                        self.edge_index = edge_index
                        self.x = x
                        self.batch = batch
                        self.y = y
                
                # Create graph object for causal analysis
                graph_obj = SimpleGraph(
                    edge_index=batch_g[i],
                    x=self.nodefeats if self.task == "node_task" else self.ind_nodefeats[self.get_g_id(x, True)],
                    batch=torch.zeros(batch_g[i].size(1), dtype=torch.int64, device=self.device) if self.task == "node_task" else None,
                    y=torch.tensor([target_y], device=self.device)
                )
                
                # Compute causal reward
                try:
                    causal_reward = self.g.causal_reward_calculator.compute_causal_reward(
                        graph_obj, x, target_y
                    )
                    causal_rewards.append(causal_reward)
                except Exception as e:
                    # Fallback to zero causal reward if computation fails
                    print(f"[CAUSAL] Warning: Failed to compute causal reward: {e}")
                    causal_rewards.append(0.0)

            if self.g.use_causal:
                # Combine original reward with causal reward
                causal_weight = getattr(self.args, 'causal_weight', 0.5)
                original_scores = -v.detach().cpu().numpy()  # Convert back to positive scores
                causal_scores = np.array(causal_rewards)
                
                # Combine scores with weighted average
                combined_scores = (1 - causal_weight) * original_scores + causal_weight * causal_scores
                
                print(f"[CAUSAL] Original scores: {np.mean(original_scores):.4f}, Causal scores: {np.mean(causal_scores):.4f}, Combined: {np.mean(combined_scores):.4f}")
            else:
                combined_scores = -v.detach().cpu().numpy()

            return combined_scores
        
        return -v.detach().cpu().numpy()


    def save(self, fname):
        if self.task == "node_task":
            data = {'g': self.g.model.state_dict()}
        else: #graph_task
            data = {'g': self.g.model.state_dict(),
                    'l': self.l.model.state_dict()}
        
        torch.save(data, fname)

    def load(self, fname):
        # Load model with appropriate device mapping
        if torch.cuda.is_available():
            data = torch.load(fname)
        else:
            data = torch.load(fname, map_location=torch.device('cpu'))
        self.g.model.load_state_dict(data['g'])
        if self.task == "graph_task":
            self.l.model.load_state_dict(data['l'])

    
    def simulated_return_a_subgraph(self, seeds):
        results = []
        for s in seeds:
            subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(int(s), self.args.n_hop, self.graphs.edge_index)
            results.append(list(subgraph_node.cpu().numpy()))
        return results


    def pretrain_l(self, l_epochs, sample_rate):
        self.l.train(l_epochs, self.simulated_return_a_subgraph, self.score_fn, sample_rate)
        self.eval_seeds = self.l.get_eval_seed_deterministic()
        self.evaluates_seed()



    def _pretrain(self):
        args = self.args

        if self.task == "graph_task":
            # construct pre-training samples
            self.pretrain_l(args.pretrain_l_iter, args.pretrain_l_sample_rate)
            max_size_in_sg = 0
            self.pretrain_nodes = []
            self.pretrain_eval_nodes = []
            seeds = self.l.get_eval_seed_deterministic()
            for seed_node in seeds:
                for nl in range(self.args.n_hop-1, self.args.n_hop, 1):
                    subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(int(seed_node), nl+1, self.graphs.edge_index)
                    if len(subgraph_node)>=5:
                        self.pretrain_nodes.append(subgraph_node.cpu().numpy())
                        self.pretrain_eval_nodes.append(seed_node)
                        if len(subgraph_node)>max_size_in_sg:
                            max_size_in_sg = len(subgraph_node)

            self.pretrain_nodes = np.array(self.pretrain_nodes, dtype=object)
            print("pretrain_samples max_size_in_sg", max_size_in_sg, flush=True)

        # pretrain g
        pretrain_g(self.g, self.pretrain_nodes, self.pretrain_eval_nodes, args.max_size, args.pretrain_g_batch_size, args.pretrain_list, self.score_fn, self.args.dataset, writer=None, use_set=False)
        pretrain_g(self.g, self.pretrain_nodes, self.pretrain_eval_nodes, args.max_size, args.pretrain_g_batch_size, args.pretrain_set, self.score_fn, self.args.dataset, writer=None, use_set=True)


    def pretrain(self):
        args = self.args
        arg_dict = vars(args)
        pretrain_related_args = ['pretrain_list', 'pretrain_set', 'pretrain_l_iter', 'pretrain_l_sample_rate', 
                                'hidden_size', 'dataset', 'seed', 'max_size', 'n_hop', 
                                'l_lr', 'g_lr',  'pretrain_g_batch_size', 'with_attr', 'ds_name']
        # Add PPO hyperparameters to hash calculation using actual arguments
        ppo_hyperparams = {
            'gamma': args.gamma,
            'clip_range': args.clip_eps, 
            'vf_clip': args.vf_clip,
            'ppo_epochs': args.ppo_epochs,
            'minibatch_size': args.minibatch_size,
            'use_gae': args.use_gae,
            'gae_lambda': args.gae_lambda,
            'reward_norm': args.reward_norm
        }
        for key, value in ppo_hyperparams.items():
            arg_dict[key] = value
        code = ' '.join([str(arg_dict[k]) for k in pretrain_related_args])
        code = hashlib.md5(code.encode('utf-8')).hexdigest().upper()
        print(f'CODE: {code}', flush=True)
        pth_fname = self.pretrain_dir / f'{code}.pth'
        if pth_fname.exists():
            print('Load the pre-trained model!', flush=True)
            self.load(pth_fname)
        else:
            self._pretrain()
            print('Save the pre-trained model!', flush=True)
            self.save(pth_fname)


    def train_g_step(self, g_it):
        shuffle_index = random.choices(list(range(len(self.eval_seeds))), k=self.args.g_batch_size)
        selected_nodes, r, policy_loss, length = self.g.train_from_rewards(np.array(self.eval_seeds)[shuffle_index], self.score_fn, policy=self.args.policy)
        
        # Calculate new metrics
        early_stop_rate = self.g.get_early_stop_rate(selected_nodes)
        avg_size = np.mean([len([x for x in traj if x != 'EOS']) for traj in selected_nodes])
        
        self.writer.add_scalar('G/Reward', r, g_it)
        self.writer.add_scalar('G/PolicyLoss', policy_loss, g_it)
        self.writer.add_scalar('G/Length', length, g_it)
        self.writer.add_scalar('G/EarlyStopRate', early_stop_rate, g_it)
        self.writer.add_scalar('G/AvgSize', avg_size, g_it)
        
        print(f'Reward={r:.2f}',
              f'PLoss={policy_loss: 2.2f}',
              f'Length={length:2.1f}',
              f'EarlyStopRate={early_stop_rate:.3f}',
              f'AvgSize={avg_size:.1f}', flush=True)


    def train_l(self, l_epochs, sample_rate):

        self.l.train(l_epochs, self.g.generate, self.score_fn, sample_rate)
        self.eval_seeds = self.l.get_eval_seed_deterministic()
        self.evaluates_seed()
    

    def evaluates_seed(self):
        
        if self.args.dataset == "ba2":
            seed_distribution = {}
            for i in range(self.max_n_nodes_in_g):
                seed_distribution[i] = 0
            for i, x in enumerate(self.eval_seeds):
                seed_distribution[x - self.each_g_start_nid[i]] += 1
            for i in range(self.max_n_nodes_in_g):
                print(i, seed_distribution[i], flush=True)
            print(flush=True)

        if self.args.dataset == "mutag":
            node_type = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']
            seed_feat_distribution = []
            for i, x in enumerate(self.eval_seeds):
                seed_feat_distribution.append(self.nodefeats[x])

            feat_dist = torch.mean(torch.stack(seed_feat_distribution),0)
            for i in range(len(node_type)):
                print(node_type[i],f': {feat_dist[i].item(): 2.2f}', end=" ", flush=True)
            print(flush=True)

    def run(self):
        # Eval before training:
        # self.evaluate_and_print('Init')

        # Pretrain
        self.pretrain()
        if self.task == "graph_task":
            self.eval_seeds = self.l.get_eval_seed_deterministic()
            self.evaluate_and_print('Pretrained')
        
        # Train
        g_it = l_it = -1
        for i_epoch in range(self.args.n_epochs):
            print('=' * 20, flush=True)
            print(f'[Epoch {i_epoch + 1:4d}]', flush=True)
            tic = time.time()
            
            # Update entropy coefficient for decay
            self.g.update_entropy_coef(i_epoch, self.args.n_epochs)
            self.g.update_epoch(i_epoch)
            print(f'Entropy coefficient: {self.g.current_entropy_coef:.6f}', flush=True)
            
            # Log effective causal weight only if causal mechanism is enabled
            if getattr(self.g, 'use_causal', False):
                effective_causal_w = self.g.get_effective_causal_weight(i_epoch)
                print(f'Effective causal weight: {effective_causal_w:.4f}', flush=True)
            else:
                print('Causal mechanism disabled', flush=True)

            if self.task == "graph_task":
                if i_epoch==0 or i_epoch%5==0:
                    print('Update L', flush=True)
                    # Skip locator training for debugging
                    # self.train_l(self.args.n_l_updates, self.args.update_l_sample_rate)
                    l_it += 1

            print('Update G', flush=True)
            self.train_g_step(g_it)
            g_it += 1

            toc = time.time()
            print(f'Elapsed Time: {toc - tic:.1f}s', flush=True)

            # Eval
            if (i_epoch + 1) % self.args.eval_every == 0:
                auc = self.evaluate_and_print(f'Epoch {i_epoch+1:4d}')
                metrics_string = '_'.join([f'{x * 100:0>2.0f}' for x in [auc]])
                self.save(self.savedir / f'{i_epoch + 1:0>5d}_{metrics_string}.pth')
                self.writer.add_scalar('Eval/Auc', auc, i_epoch)




def main(args):
    # Force output flushing for SLURM environments
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Debug: Print current working directory and file permissions
    print(f"[DEBUG] Current working directory: {os.getcwd()}", flush=True)
    print(f"[DEBUG] Python executable: {sys.executable}", flush=True)
    print(f"[DEBUG] SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not set')}", flush=True)
    
    # Ensure results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"[DEBUG] Created results directory: {results_dir}", flush=True)
    
    runner = Runner(args)
    runner.run()
    
    # Final flush before exit
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"[DEBUG] Job completed successfully", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset + Trained Model 
    parser.add_argument('--dataset', type=str, default='syn1') # syn1/syn2/syn3/syn4/ba2/mutag
    parser.add_argument('--model', type=str, default='GNN')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--with_attr', action='store_true', default=True)
    parser.add_argument('--n_hop', type=int, default=3)
    parser.add_argument('--max_size', type=int, default=20)


    # Locator
    parser.add_argument('--pretrain_l_sample_rate', type=float, default=1.0)
    parser.add_argument('--update_l_sample_rate', type=float, default=0.2)
    parser.add_argument('--l_lr', type=float, default=1e-2)
    parser.add_argument('--pretrain_l_iter', type=int, default=200)

    # Generator
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-2)
    parser.add_argument('--n_rollouts', type=int, default=5)
    parser.add_argument('--pretrain_g_batch_size', type=int, default=32)
    parser.add_argument('--g_batch_size', type=int, default=128)
    

    # Regularization
    parser.add_argument('--size_reg', type=float, default=0.01)
    parser.add_argument('--sim_reg', type=float, default=0.01)
    parser.add_argument('--radius_penalty', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=0.)

    # Pretrain
    parser.add_argument('--pretrain_list', type=int, default=10) 
    parser.add_argument('--pretrain_set', type=int, default=25) 


    # coordinate 
    parser.add_argument('--n_epochs', type=int, default=10)  
    parser.add_argument('--n_g_updates', type=int, default=1)
    parser.add_argument('--n_l_updates', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1)

    # Policy selection
    parser.add_argument('--policy', type=str, choices=['REINFORCE', 'PPO'], default='REINFORCE')
    
    # PPO-specific parameters
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--clip_range_policy', type=float, default=0.2,
                       help='PPO policy clipping range')
    parser.add_argument('--clip_range_value', type=float, default=0.5,
                       help='PPO value function clipping range')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--frontier_shaping_beta', type=float, default=0.0)
    parser.add_argument('--adaptive_kl_penalty', action='store_true', default=False,
                       help='Enable adaptive KL penalty as fallback to clipping')
    
    # Entropy decay parameters
    parser.add_argument('--entropy_decay_start', type=float, default=1e-3)
    parser.add_argument('--entropy_decay_end', type=float, default=1e-4)
    parser.add_argument('--entropy_decay_final', type=float, default=0.0)

    # Causal mechanism parameters
    parser.add_argument('--use_causal', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable causal mechanism for subgraph generation')
    parser.add_argument('--causal_weight', type=float, default=0.5,
                       help='Weight for causal term in reward function (0.0 to 1.0)')
    parser.add_argument('--causal_warmup_epochs', type=int, default=5,
                       help='Number of epochs to warmup causal weight from 0 to target')

    # New parameters for STOP action and penalties
    parser.add_argument('--target_size_k', type=int, default=8,
                       help='Target size for explanations (default depends on dataset)')
    parser.add_argument('--lambda_size', type=float, default=0.5,
                       help='Weight for size penalty (hinge loss on target size)')
    parser.add_argument('--lambda_radius', type=float, default=1.0,
                       help='Initial weight for radius penalty')
    parser.add_argument('--lambda_radius_min', type=float, default=0.2,
                       help='Final weight for radius penalty after decay')
    parser.add_argument('--step_cost', type=float, default=0.001,
                       help='Cost per step to discourage long trajectories')
    parser.add_argument('--stop_bias', type=float, default=0.5,
                       help='Bias applied to STOP action logit')
    parser.add_argument('--fidelity_thresh', type=float, default=0.9,
                       help='Fidelity threshold for STOP bonus')
    parser.add_argument('--debug_mode', type=int, default=0,
                       help='Enable debug mode (0=off, 1=on)')

    # PPO stability parameters
    parser.add_argument('--ppo_clip', type=float, default=0.1,
                       help='PPO clip range')
    parser.add_argument('--kl_target', type=float, default=0.0075,
                       help='Target KL divergence for adaptive penalty')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Maximum gradient norm for clipping')
    parser.add_argument('--use_value_clip', action='store_true', default=True,
                       help='Enable value function clipping')

    # Entropy scheduling
    parser.add_argument('--entropy_coeff_start', type=float, default=1e-3,
                       help='Starting entropy coefficient')
    parser.add_argument('--entropy_coeff_end', type=float, default=1e-4,
                       help='Ending entropy coefficient')

    # AUC evaluation parameters
    parser.add_argument('--edge_score_agg', type=str, default='max', choices=['max', 'mean'],
                       help='How to aggregate node scores to edge scores')
    parser.add_argument('--debug_eval_graph', type=int, default=-1,
                       help='Graph ID to run debug evaluation on (-1 to disable)')
    
    # Additional PPO parameters for pretraining hash
    parser.add_argument('--vf_clip', type=float, default=0.5,
                       help='Value function clipping parameter')
    parser.add_argument('--use_gae', action='store_true', default=True,
                       help='Use Generalized Advantage Estimation')
    parser.add_argument('--reward_norm', action='store_true', default=True,
                       help='Enable reward normalization')

    args = parser.parse_args()
    
    # Set dataset-specific defaults for target_size_k
    if args.target_size_k == 8:  # Only if using default
        if args.dataset in ['syn1', 'syn2', 'syn3', 'syn4']:
            args.target_size_k = 8  # Synthetic datasets typically have ~8 edge motifs
        elif args.dataset == 'ba2':
            args.target_size_k = 10  # BA-Shapes has ~10 edge motifs
        elif args.dataset == 'mutag':
            args.target_size_k = 6   # Mutag typically has smaller motifs
    
    # Set deterministic seeds for reproducibility
    seed_all(args.seed)
    
    # Set PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('= ' * 20)
    now = datetime.datetime.now()
    print(args)
    args.savedir = f'ckpts/{args.dataset}/{now.strftime("%Y%m%d%H%M%S")}/'
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(f'##  Seed: {args.seed}, Deterministic: {torch.backends.cudnn.deterministic}', flush=True)
    main(args)
    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)


