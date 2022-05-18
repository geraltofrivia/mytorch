"""
    One stop shop for most of your metrics needs
"""
import re
from functools import partial
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
        return torch.mean(torch.nonzero((torch.argsort(-preds, dim=1) == target).to(torch.int))[:, 1] + 1.0)

    @staticmethod
    def mean_reciprocal_rank(preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """ Average reciprocal rank. Preds: (n_instances, m_classes), target: (n_instance, 1) """
        ranks = torch.nonzero((torch.argsort(-preds, dim=1) == target).to(torch.int))[:, 1] + 1.0
        return torch.mean(1 / ranks)

    @staticmethod
    def hits_at(preds: torch.Tensor, target: torch.Tensor, k: int, *args, **kwargs) -> torch.Tensor:
        """ Hits at K, Preds: (n_instances, m_classes), target: (n_instance, 1) """

        # assert preds.shape[1] >= k, f"K is too high for a tensor of shape {preds.shape}"
        return (torch.argsort(-preds, dim=1)[:, :k] == target).any(dim=1).to(torch.float).mean()


class MetricsWrapper:
    """
        Usage:

            > mc = MetricsWrapper.from_args(('acc', 'mr'))
            > mc(target=target, preds=preds)
            {'acc': 0.25, 'mr': 307.5}

    """

    def __init__(self, metric_nms: List[str], metric_fns: List[Callable]):
        self.metric_nms = metric_nms
        self.metric_fns = metric_fns

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        if len(target.shape) == 1:
            target = target.unsqueeze(1)

        return {
            nm: float(f"{fn(preds=preds, target=target).item():.5f}")
            for nm, fn in zip(self.metric_nms, self.metric_fns)
        }

    def __repr__(self):
        return f"{type(self)} object containing the following metrics - {self.metric_nms}"

    @classmethod
    def from_args(cls, args: List[str]):
        torchmetrics_metrics = {
            'acc': 'accuracy',
            'accuracy': 'accuracy',
        }

        local_metrics = {
            'hitsat': 'hits_at',
            'hitsat_': 'hits_at',
            'h': 'hits_at',
            'hits@': 'hits_at',
            'hits_at': 'hits_at',
            'hits_at_': 'hits_at',
            'mr': 'mean_rank',
            'meanrank': 'mean_rank',
            'mean_rank': 'mean_rank',
            'reciprocal_rank': 'mean_reciprocal_rank',
            'mrr': 'mean_reciprocal_rank',
            'mean_reciprocal_rank': 'mean_reciprocal_rank',
            'retrieval_reciprocal_rank': 'mean_reciprocal_rank'
        }

        callables: List[Callable] = []

        for arg in args:

            if arg.startswith('h') and re.search(r'\d+$', arg) is not None:
                suffix = re.search(r'\d+$', arg).group()
                arg = arg.replace(suffix, '')
                k = int(suffix)
            else:
                k = None

            # Find if the arg is known in torchmetrics (based on dict above)
            if arg in torchmetrics_metrics:
                callables.append(getattr(tm.functional, torchmetrics_metrics[arg]))

            # Find if the arg is known in local metrics implemented in this file
            elif arg in local_metrics:
                callables.append(partial(getattr(MultiClassSingleLabelMetrics, local_metrics[arg]), k=k))

            # Raise MetricNotUnderstoodError
            else:
                raise UnknownMetricName(f"Metric invoked by the name {arg} is not understood.")
            ...

        return cls(args, callables)
