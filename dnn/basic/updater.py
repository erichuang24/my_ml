# -*- coding: utf-8 -*-
import numpy as np
from utils.data_utils import *
from utils.gradient_check import *

"""

频繁使用的神经网络一阶梯度更新规则。每次更新接收：当前的网络权重，
训练获得的梯度，以及相关配置进行权重更新。
def update(w, dw, config = None):
Inputs:
  - w:当前权重.
  - dw: 和权重形状相同的梯度.
  - config: 字典型超参数配置，比如学习率，动量值等。如果更新规则需要用到缓存，
    在配置中需要保存相应的缓存。
Returns:
  - next_w: 更新后的权重.
  - config: 更新规则相应的配置.
"""


# class updater:
def sgd(w, dw, config=None):
    """
    随机梯度下降更新规则.

    config :
    - learning_rate: 学习率.
    """
    if config is None:
        config = {}
        config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    动量随机梯度下降更新规则。
    config 使用格式:
    - learning_rate: 学习率。
    - momentum: [0,1]的动量衰减因子，0表示不使用动量，即退化为SGD。
    - velocity: 和w，dw形状相同的速度。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.setdefault('velocity', np.zeros_like(w))

    next_w = None
    #############################################################################
    #                       任务：实现动量更新                                  #
    #         更新后的速度存放在v中，更新后的权重存放在next_w中                 #
    #############################################################################
    # v = config['momentum'] * config['velocity'] - config['learning_rate'] * dw
    v = config['momentum'] * config['velocity'] + (1 - config['momentum']) * dw

    next_w = w - config['learning_rate'] * v
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    RMSProp更新规则

    config 使用格式:
    - learning_rate: 学习率.
    - decay_rate:历史累积梯度衰减率因子,取值为[0,1]
    - epsilon: 避免除零异常的小数.
    - cache:历史梯度缓存.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    #############################################################################
    #                         任务：实现 RMSprop 更新                           #
    #  将更新后的权重保存在next_w中，将历史梯度累积存放在config['cache']中。    #
    #############################################################################
    s = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw * dw
    next_w = w - config['learning_rate'] * dw / (np.sqrt(s) + config['epsilon'])
    config['cache'] = s
    #############################################################################
    #                             结束编码                                      #
    #############################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    使用 Adam更新规则 ,融合了“热身”更新操作。

    config 使用格式:
    - learning_rate: 学习率.
    - beta1: 动量衰减因子.
    - beta2: 学习率衰减因子.
    - epsilon: 防除0小数.
    - m: 梯度.
    - v: 梯度平方.
    - t: 迭代次数.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    #############################################################################
    #                          任务：实现Adam更新                               #
    #     将更新后的权重存放在next_w中，记得将m,v,t存放在相应的config中         #
    #############################################################################
    config['t'] = config['t'] + 1

    v = config['v'] * config['beta1'] + (1 - config['beta1']) * dw
    s = config['m'] * config['beta2'] + (1 - config['beta2']) * dw * dw

    v = v / (1 - np.power(config['beta1'], config['t']))
    s = s / (1 - np.power(config['beta2'], config['t']))

    next_w = w - config['learning_rate'] * v / (np.sqrt(s) + config['epsilon'])

    config['v'] = v
    config['w'] = w

    #############################################################################
    #                            结束编码                                       #
    #############################################################################

    return next_w, config  # N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
# m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
# v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)
#
# u = updater()
# config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
# next_w, _ = u.adam(w, dw, config=config)
#
# expected_next_w = np.asarray([
#     [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
#     [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
#     [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
#     [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
# expected_v = np.asarray([
#     [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853, ],
#     [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385, ],
#     [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767, ],
#     [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966, ]])
# expected_m = np.asarray([
#     [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
#     [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
#     [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
#     [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])
#
# print('更新权重误差: ', rel_error(expected_next_w, next_w))
# print('速度误差: ', rel_error(expected_v, config['v']))
# print('速度误差: ', rel_error(expected_m, config['m']))
