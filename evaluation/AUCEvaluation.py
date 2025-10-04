from evaluation.BaseEvaluation import BaseEvaluation
from evaluation.utils import evaluation_auc, debug_eval_one


class AUCEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.
    
    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.
    
    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, task, ground_truth, indices):
        self.task = task
        self.ground_truth = ground_truth
        self.indices = indices

    def get_score(self, explanations, edge_score_agg='max', debug_eval_graph=-1):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :param edge_score_agg: How to aggregate node scores to edge scores ('max' or 'mean')
        :param debug_eval_graph: If >= 0, run debug evaluation for single graph
        :return: auc score
        """
        # Run debug evaluation if requested
        if debug_eval_graph >= 0:
            debug_result = debug_eval_one(debug_eval_graph, explanations, self.ground_truth, self.task, edge_score_agg)
            if debug_result is not None:
                print(f"Debug evaluation completed for graph {debug_eval_graph}")
        
        return evaluation_auc(self.task, explanations, self.ground_truth, self.indices, edge_score_agg, debug_eval_graph)
