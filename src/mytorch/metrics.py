"""
    One stop shop for most of your metrics needs
"""
import torch
import numpy as np

from utils.goodies import InconsistentType

class MultiClassUniLabelMetrics(object):
    """
        Collection of commonly used metrics for multi class, uni label problems (most problems including MLM, LinkPred)
    """

    @staticmethod
    def acc(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Accuracy. true: (n,1); pred: (n, m); return float"""
        if type(true) is torch.Tensor and type(pred) is torch.Tensor:
            return MultiClassUniLabelMetrics._torch_acc_(true=true, pred=pred)
        elif type(true) is np.ndarray and type(pred) is np.ndarray:
            return MultiClassUniLabelMetrics._numpy_acc_(true=true, pred=pred)
        else:
            raise InconsistentType("True and Pred belong to unknown/different types)")

    @staticmethod
    def _torch_acc_(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Accuracy. true: (n,1); pred: (n, m); return float"""
        return torch.mean((torch.argmax(pred, dim=1) == true).to(torch.float)).item()

    @staticmethod
    def _numpy_acc_(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Accuracy. true: (n,1); pred: (n, m); return float"""
        ...

    @staticmethod
    def mr(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Mean Rank. true: (n,1); pred: (n, m); return float"""
        if type(true) is torch.Tensor and type(pred) is torch.Tensor:
            return MultiClassUniLabelMetrics._torch_mr_(true=true, pred=pred)
        elif type(true) is np.ndarray and type(pred) is np.ndarray:
            return MultiClassUniLabelMetrics._numpy_mr_(true=true, pred=pred)
        else:
            raise InconsistentType("True and Pred belong to unknown/different types)")

    @staticmethod
    def _torch_mr_(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Mean Rank. true: (n,1); pred: (n, m); return float """
        ...

    @staticmethod
    def _numpy_mr_(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Mean Rank. true: (n,1); pred: (n, m); return float """
        ...

    @staticmethod
    def mrr(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Mean Reciprocal Rank. true: (n,1); pred: (n, m); return float"""
        if type(true) is torch.Tensor and type(pred) is torch.Tensor:
            return MultiClassUniLabelMetrics._torch_mrr_(true=true, pred=pred)
        elif type(true) is np.ndarray and type(pred) is np.ndarray:
            return MultiClassUniLabelMetrics._numpy_mrr_(true=true, pred=pred)
        else:
            raise InconsistentType("True and Pred belong to unknown/different types)")

    @staticmethod
    def _numpy_mrr_(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Mean Reciprocal Rank. true: (n,1); pred: (n, m); return float """
        ...

    @staticmethod
    def _torch_mrr_(true: torch.Tensor, pred: torch.Tensor) -> float:
        """ Mean Reciprocal Rank. true: (n,1); pred: (n, m); return float """
        ...

    @staticmethod
    def hits_at(true: torch.Tensor, pred: torch.Tensor, k: int) -> float:
        """ Mean Rank. true: (n,1); pred: (n, m); return float"""
        if type(true) is torch.Tensor and type(pred) is torch.Tensor:
            return MultiClassUniLabelMetrics._torch_hits_at_(true=true, pred=pred, k=k)
        elif type(true) is np.ndarray and type(pred) is np.ndarray:
            return MultiClassUniLabelMetrics._numpy_hits_at_(true=true, pred=pred, k=k)
        else:
            raise InconsistentType("True and Pred belong to unknown/different types)")

    @staticmethod
    def _torch_hits_at_(true: torch.Tensor, pred: torch.Tensor, k: int) -> float:
        """ Hits at K, true: (n,1); pred: (n, m); return float """
        ...

    @staticmethod
    def _numpy_hits_at_(true: torch.Tensor, pred: torch.Tensor, k: int) -> float:
        """ Hits at K, true: (n,1); pred: (n, m); return float """
        ...
    