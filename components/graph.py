from typing import Union, Optional, List, Set, Dict
import collections

import numpy as np
from scipy import sparse as sp
import random
import torch

class Graph:

    def __init__(self, edges):
        self.edge_index = torch.tensor(edges)
        edges = edges.T
        self.neighbors, self.n_nodes, self.adj_mat, self.isolated_nodes = self._init_from_edges(edges)

    @classmethod
    def from_dropped_edges(cls, original_graph, drop_prob, preserve_connectivity=False):
        """
        Create a new Graph with edges randomly dropped.
        
        This factory method applies structural data augmentation by randomly dropping
        edges during training. Edge dropout modifies world dynamics (action availability,
        neighborhood expansion, subgraph construction) but does NOT affect reward computation.
        
        Args:
            original_graph: Original Graph object
            drop_prob: float in [0, 1], probability of dropping each edge
            preserve_connectivity: bool, whether to preserve graph connectivity
        
        Returns:
            Graph object with dropped edges (or original if drop_prob <= 0)
        
        Example:
            >>> original = Graph(edges)
            >>> dropped = Graph.from_dropped_edges(original, drop_prob=0.2)
            >>> dropped.edge_index.size(1) <= original.edge_index.size(1)
            True
        """
        from utils.edge_dropout import apply_edge_dropout
        
        if drop_prob <= 0.0:
            return original_graph
        
        # Apply edge dropout
        dropped_edge_index = apply_edge_dropout(
            original_graph.edge_index,
            drop_prob,
            preserve_connectivity,
            device=original_graph.edge_index.device
        )
        
        # Handle edge case: if all edges were dropped, return original graph
        if dropped_edge_index.size(1) == 0:
            return original_graph
        
        # Create new Graph from dropped edges
        return cls(dropped_edge_index.cpu().numpy())

    @staticmethod
    def _init_from_edges(edges: np.ndarray) -> (Dict[int, Set[int]], int, sp.spmatrix):
        neighbors = collections.defaultdict(set)
        max_id = -1
        for u, v in edges:
            max_id = max(max_id, u, v)
            if u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
        n_nodes = len(neighbors)
        isolated_nodes = []
        if (max_id + 1) != n_nodes:
            print("There exists isolated nodes! Some nodes do not have edges")
            for i in range(max_id + 1):
                if i not in neighbors:
                    isolated_nodes.append(i)

        n_nodes = max_id + 1
        adj_mat = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(max_id + 1, max_id + 1))
        adj_mat += adj_mat.T
        isolated_nodes = set(isolated_nodes)
        return neighbors, n_nodes, adj_mat, isolated_nodes

    def outer_boundary(self, nodes: Union[List, Set]) -> Set[int]:
        boundary = set()
        for u in nodes:
            boundary |= self.neighbors[u]
        boundary.difference_update(nodes)
        return boundary

    def k_ego(self, nodes: Union[List, Set], k: int) -> Set[int]:
        ego_nodes = set(nodes)
        current_boundary = set(nodes)
        for _ in range(k):
            current_boundary = self.outer_boundary(current_boundary) - ego_nodes
            ego_nodes |= current_boundary
        return ego_nodes

    def check_valid_expansion(self, expansion: List[int]) -> bool:
        if len(expansion) != len(set(expansion)):
            return False
        for i in range(1, len(expansion) - 1):
            boundary = self.outer_boundary(expansion[:i])
            if expansion[i] not in boundary:
                return False
        return True
    
    def sample_expansion_with_high_scores(self, score_fn, sample_num, comm_nodes, seed, max_size):

        sample_walks = [self.sample_layerwise_expansion(comm_nodes, seed, max_size) for s in range(sample_num)]
        scores = score_fn(sample_walks)
        return sample_walks[np.argmax(scores)]

    def sample_expansion_from_community(self, comm_nodes: Union[List, Set],
                                        seed: int, max_size: int) -> List[int]:
        if seed is None:
            seed = random.choice(tuple(comm_nodes))
        remaining = set(comm_nodes) - {seed}
        boundary = self.neighbors[seed].copy()
        walk = [seed]
        while len(remaining):
            candidates = tuple(boundary & remaining)
            
            # Handle disconnected graph (e.g., from edge dropping)
            if not candidates:
                # Graph is disconnected - return partial walk
                break
                
            new_node = random.choice(candidates)
            remaining.remove(new_node)
            boundary |= self.neighbors[new_node]
            walk.append(new_node)
        
        length = np.random.randint(4, len(walk)) if len(walk) >= 4 else len(walk)
        return walk[:length]
    
    def sample_layerwise_expansion(self, comm_nodes: Union[List, Set],
                                        seed: int, max_size: int) -> List[int]:

        remaining = set(comm_nodes) - {seed}
        cur_boundary = self.neighbors[seed].copy()
        next_boundary = None
        walk = [seed]
        while len(remaining):
            candidates = tuple(cur_boundary & remaining)
            
            # Handle disconnected graph (e.g., from edge dropping)
            if not candidates:
                # Try to expand to next boundary
                if next_boundary is not None:
                    cur_boundary = next_boundary & remaining
                    next_boundary = None
                    candidates = tuple(cur_boundary & remaining)
                
                # If still no candidates, graph is disconnected - return partial walk
                if not candidates:
                    break
            
            new_node = random.choice(candidates)
            remaining.remove(new_node)
            cur_boundary.remove(new_node)
            if next_boundary is None:
                next_boundary = self.neighbors[new_node].copy()
            else:
                next_boundary |= self.neighbors[new_node]
            
            if len(cur_boundary)==0:
                cur_boundary = next_boundary & remaining
                next_boundary = None
            
            walk.append(new_node)
        
        length = np.random.randint(4, len(walk)) if len(walk) >= 4 else len(walk)

        return walk[:length]


    def sample_expansion(self, max_size: int, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = random.randint(0, self.n_nodes - 1)
        walk = [seed]
        boundary = self.neighbors[seed].copy()
        for i in range(max_size - 1):
            candidates = tuple(boundary - set(walk))
            
            # Handle disconnected graph or exhausted boundary
            if not candidates:
                break
                
            new_node = random.choice(candidates)
            boundary |= self.neighbors[new_node]
            walk.append(new_node)
        return walk

    def connected_components(self, nodes):
        remaining = set(nodes)
        ccs = []
        cc = set()
        queue = collections.deque()
        while len(remaining) or len(queue):
            # print(queue, remaining)
            if len(queue) == 0:
                if len(cc):
                    ccs.append(cc)
                v = remaining.pop()
                cc = {v}
                queue.extend(self.neighbors[v] & remaining)
                remaining -= {v}
                remaining -= self.neighbors[v]
            else:
                v = queue.popleft()
                queue.extend(self.neighbors[v] & remaining)
                cc |= (self.neighbors[v] & remaining) | {v}
                remaining -= self.neighbors[v]
        if len(cc):
            ccs.append(cc)
        return ccs

    def subgraph_depth(self, nodes, seed=None):
        if seed is None:
            seed = nodes[0]
        remaining = set(nodes)
        remaining.remove(seed)
        q = collections.deque()
        q.append((seed, 0))
        while len(q):
            u, depth = q.popleft()
            neighbors = self.neighbors[u] & remaining
            q.extend([(v, depth + 1) for v in neighbors])
            remaining -= neighbors
        return depth
