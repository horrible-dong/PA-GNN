# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['DefaultEvaluator']

import itertools
import warnings
from typing import List

from sklearn import metrics as sklearn_metrics

from ..utils.misc import all_gather

warnings.filterwarnings('ignore')


class DefaultEvaluator:
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.probs = []
        self.outputs = []
        self.targets = []
        self.eval = {metric: None for metric in metrics}

    def update(self, outputs, targets):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'update(self, outputs, targets)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs['logits']
        self.probs += outputs.softmax(1)[:, 1].tolist()
        outputs = outputs.max(1)[1].tolist()
        targets = targets.tolist()
        self.outputs += outputs
        self.targets += targets

    def synchronize_between_processes(self):
        self.probs = list(itertools.chain(*all_gather(self.probs)))
        self.outputs = list(itertools.chain(*all_gather(self.outputs)))
        self.targets = list(itertools.chain(*all_gather(self.targets)))

    @staticmethod
    def metric_acc(**kwargs):
        return sklearn_metrics.accuracy_score(kwargs['targets'], kwargs['outputs'])

    @staticmethod
    def metric_recall(**kwargs):
        return sklearn_metrics.recall_score(kwargs['targets'], kwargs['outputs'], average='macro')

    @staticmethod
    def metric_precision(**kwargs):
        return sklearn_metrics.precision_score(kwargs['targets'], kwargs['outputs'], average='macro')

    @staticmethod
    def metric_f1(**kwargs):
        return sklearn_metrics.f1_score(kwargs['targets'], kwargs['outputs'], average='macro')

    @staticmethod
    def metric_auc(**kwargs):
        return sklearn_metrics.roc_auc_score(kwargs['targets'], kwargs['probs'], average='macro')

    @staticmethod
    def metric_ap(**kwargs):
        return sklearn_metrics.average_precision_score(kwargs['targets'], kwargs['probs'], average='macro')

    def summarize(self):
        print('Classification Metrics:')
        for metric in self.metrics:
            value = getattr(self, f'metric_{metric}')(probs=self.probs, outputs=self.outputs, targets=self.targets)
            self.eval[metric] = value
            print(f'{metric}: {value * 100:.2f}', end='    ')
        print('\n')
