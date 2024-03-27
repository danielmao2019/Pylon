from typing import List, Dict
import torch


class ConfusionMatrix:

    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.reset()

    @staticmethod
    def _average_(scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        r"""This function adds a new field to the dictionary holding the average scores across all tasks.
        """
        assert 'average' not in scores.keys()
        average: Dict[str, List[float]] = {}
        for task in scores.keys():
            for metric, val in scores[task].items():
                average[metric] = average.get(metric, []) + [val]
        for metric, vals in average.items():
            average[metric]: float = sum(vals) / len(vals)
        scores['average'] = average
        return scores

    def update(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        comparable_keys = set(y_pred.keys()) & set(y_true.keys())
        assert set(self.label_names).issubset(comparable_keys)
        batch_size = list(y_pred.values())[0].shape[0]
        result: Dict[str, Dict[str, float]] = dict((key, {}) for key in comparable_keys)
        # compute and update metrics
        for key in comparable_keys:
            assert y_pred[key].shape[0] == y_true[key].shape[0] == batch_size
            # compute batch statistics
            batch_tp = torch.sum(torch.logical_and(y_pred[key]>=0.5, y_true[key]==1).type(torch.int32))
            batch_tn = torch.sum(torch.logical_and(y_pred[key]<0.5, y_true[key]==0).type(torch.int32))
            batch_fp = torch.sum(torch.logical_and(y_pred[key]>=0.5, y_true[key]==0).type(torch.int32))
            batch_fn = torch.sum(torch.logical_and(y_pred[key]<0.5, y_true[key]==1).type(torch.int32))
            assert batch_tp + batch_tn + batch_fp + batch_fn == batch_size, \
                f"{batch_tp=}, {batch_tn=}, {batch_fp=}, {batch_fn=}, {batch_size=}, {y_pred[key]=}, {y_true[key]=}"
            # update batch result
            result[key]['tp'] = batch_tp
            result[key]['tn'] = batch_tn
            result[key]['fp'] = batch_fp
            result[key]['fn'] = batch_fn
            # update aggregate result
            self.count[key]['tp'] += batch_tp
            self.count[key]['tn'] += batch_tn
            self.count[key]['fp'] += batch_fp
            self.count[key]['fn'] += batch_fn
        result = self._average_(result)
        return result

    def reset(self):
        self.count: Dict[str, Dict[str, float]] = dict(
            (key, {
                'tp': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0,
            }) for key in self.label_names
        )

    def get_accuracy(self, task: str) -> float:
        count = self.count[task]
        accuracy = (count['tp'] + count['tn']) / (count['tp'] + count['tn'] + count['fp'] + count['fn'])
        return accuracy.item()

    def get_precision(self, task: str) -> float:
        count = self.count[task]
        precision = count['tp'] / (count['tp'] + count['fp'])
        return precision.item()

    def get_recall(self, task: str) -> float:
        count = self.count[task]
        recall = count['tp'] / (count['tp'] + count['fn'])
        return recall.item()

    def get_f1(self, task: str) -> float:
        count = self.count[task]
        f1 = 2 * count['tp'] / (2 * count['tp'] + count['fp'] + count['fn'])
        return f1.item()

    def get_scores(self) -> Dict[str, Dict[str, float]]:
        result = dict(
            (task, {
                'accuracy': self.get_accuracy(task=task),
                'precision': self.get_precision(task=task),
                'recall': self.get_recall(task=task),
                'f1': self.get_f1(task=task),
            }) for task in self.label_names
        )
        result = self._average_(result)
        return result
