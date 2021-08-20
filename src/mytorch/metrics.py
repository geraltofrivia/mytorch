"""
    One stop shop for most of your metrics needs
"""
from typing import List, Callable

import torch
import torchmetrics as tm

from .utils.goodies import UnknownMetricName


class MultiClassSingleLabelMetrics(object):
    """
        Collection of commonly used metrics for multi class, uni label problems (most problems including MLM, LinkPred)
    """

    @staticmethod
    def mean_rank(preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Average rank. Preds: (n_instances, m_classes), target: (n_instance, 1) """
        return torch.mean(torch.nonzero((torch.argsort(-preds, dim=1) == target).to(torch.int))[:,1]+1.0)

    @staticmethod
    def hits_at(preds: torch.Tensor, target: torch.Tensor, k: int, *args, **kwargs) -> torch.Tensor:
        """ Hits at K, Preds: (n_instances, m_classes), target: (n_instance, 1) """

        assert preds.shape[1] >= k, f"K is too high for a tensor of shape {preds.shape}"
        return (torch.argsort(-preds, dim=1)[:,:5] == target).any(dim=1).to(torch.float).mean()


class MetricsWrapper:

    def __init__(self, metric_nms: List[str], metric_fns: List[Callable]):
        self.metric_nms = metric_nms
        self.metric_fns = metric_fns

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, average: str = 'micro'):
        if len(target.shape) == 1:
            target = target.unsqueeze(1)

        return {
            nm: fn(preds=preds, target=target, average=average).item()
            for nm, fn in zip(self.metric_nms, self.metric_fns)
        }

    def __repr__(self):
        return f"{type(self)} object containing the following metrics - {self.metric_nms}"

    @classmethod
    def from_args(cls, args: List[str]):
        torchmetrics_metrics = {
            'acc': 'accuracy',
            'accuracy': 'accuracy',
            'reciprocal_rank': 'retrieval_reciprocal_rank',
            'mrr': 'retrieval_reciprocal_rank',
            'mean_reciprocal_rank': 'retrieval_reciprocal_rank',
            'retrieval_reciprocal_rank': 'retrieval_reciprocal_rank'
        }

        local_metrics = {
            'hitsat':'hits_at',
            'hits@': 'hits_at',
            'hits_at': 'hits_at',
            'mr': 'mean_rank',
            'meanrank': 'mean_rank',
            'mean_rank': 'mean_rank'
        }

        callables: List[Callable] = []

        for arg in args:

            # Find if the arg is known in torchmetrics (based on dict above)
            if arg in torchmetrics_metrics:
                callables.append(getattr(tm.functional, torchmetrics_metrics[arg]))

            # Find if the arg is known in local metrics implemented in this file
            elif arg in local_metrics:
                callables.append(getattr(MultiClassSingleLabelMetrics, local_metrics[arg]))

            # Raise MetricNotUnderstoodError
            else:
                raise UnknownMetricName(f"Metric invoked by the name {arg} is not understood.")
            ...

        return cls(args, callables)
