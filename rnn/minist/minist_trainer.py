# -*- coding: utf-8 -*-
import numpy as np

from rnn.minist_utils import *
import dnn.basic.updater


class MinistTrainer(object):
    """
    CaptioningTrainer大部分内容和前面的Trainer相同
    使用方法：
    data = load_coco_data()
    model = MyAwesomeModel(hidden_dim=100)
    trainer = CaptioningTrainer(model, data,
                    update_rule='sgd',
                    updater_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    trainer.train()
    """

    def __init__(self, model, data, **kwargs):
        """
        初始化CaptioningTrainer
        所需参数:
        - model: RNN模型
        - data: coco数据集

        可选参数:
        - update_rule:更新规则，查看 updater.py.
          默认为 'sgd'.
        - updater_config: 更新器配置
        - lr_decay:学习率衰减因子
        - batch_size: 批量大小
        - num_epochs: 迭代次数
        - print_every:每训练多少次，打印训练结果
        - verbose:是否打印训练中间结果
        """
        self.model = model
        self.data = data

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.updater_config = kwargs.pop('updater_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        if not hasattr(dnn.updater, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(dnn.updater, self.update_rule)

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.updater_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.updater_config.items()}
            self.updater_configs[p] = d

    def _step(self):
        minibatch = sample_mnist_minibatch(self.data['images'],
                                           self.data['labels'],
                                           batch_size=self.batch_size)
        features, label = minibatch

        loss, grads = self.model.loss(features, label)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.updater_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.updater_configs[p] = next_config

    def train(self):
        num_train = self.data['labels'].shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(int(num_iterations)):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.updater_configs:
                    self.updater_configs[k]['learning_rate'] *= self.lr_decay
