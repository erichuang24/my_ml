# -*- coding: utf-8 -*-
import numpy as np


def affine_forward(x, w, b):
    """
    计算神经网络当前层的前馈传播，该方法计算在全连接情况下的得分函数。
    注：如果不理解affine仿射变换，简单的理解为在全连接情况下的得分函数即可。

    输入数据x的形状为(N, d_1, ..., d_k)，其中N表示数据量，(d_1, ..., d_k)表示
    每一通道的数据维度，如果是图片数据就为(长，宽，色道)。数据的总维度为
    D = d_1 * ... * d_k，因此我们需要将数据重塑成形状为(N,D)的数组再进行仿射变换。

    Inputs:
    - x: 输入数据，其形状为(N, d_1, ..., d_k)的numpy数组。
    - w: 权重矩阵，其形状为(D,M)的numpy数组，D表示输入数据维度，M表示输出数据维度
         可以将D看成输入的神经元个数，M看成输出神经元个数。
    - b: 偏置向量，其形状为(M,)的numpy数组。

    Returns 元组:
    - out: 形状为(N, M)的输出结果。
    - cache: 将输入进行缓存(x, w, b)。
    """
    out = None
    #############################################################################
    #                      任务: 实现全连接前向传播                             #
    #                   注：首先你需要将输入数据重塑成行。                      #
    #############################################################################
    N = x.shape[0]
    x_new = x.reshape(N, -1)
    out = np.dot(x_new, w) + b

    #############################################################################
    #                             结束编码                                      #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
   计算仿射层的反向传播.

    Inputs:
    - dout: 形状为(N, M)的上层梯度
    - cache: 元组:
      - x: (N, d_1, ... d_k)的输入数据
      - w: 形状为(D, M)的权重矩阵

    Returns 元组:
    - dx: 输入数据x的梯度，其形状为(N, d1, ..., d_k)
    - dw: 权重矩阵w的梯度，其形状为(D,M)
    - db: 偏置项b的梯度，其形状为(M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    #                  任务: 实现仿射层反向传播                                 #
    #         注意：你需要将x重塑成(N,D)后才能计算各梯度，                      #
    #              求完梯度后你需要将dx的形状与x重塑成一样                      #
    #############################################################################
    N = x.shape[0]
    x_new = x.reshape(N, -1)
    dw = np.dot(x_new.T, dout)
    dx_new = np.dot(dout, w.T)
    dx = np.reshape(dx_new, x.shape)
    db = np.sum(dout, axis=0)

    #############################################################################
    #                             结束编码                                      #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    计算rectified linear units (ReLUs)激活函数的前向传播，并保存相应缓存

    Input:
    - x: 输入数据

    Returns 元组:
    - out: 和输入数据x形状相同
    - cache: x
    """
    out = None
    #############################################################################
    #             任务: 实现ReLU 的前向传播.                                    #
    #            注意：你只需要1行代码即可完成                                  #
    #############################################################################
    out = np.maximum(0, x)

    #############################################################################
    #                             结束编码                                      #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    计算 rectified linear units (ReLUs)激活函数的反向传播.

    Input:
    - dout: 上层误差梯度
    - cache: 输入 x,其形状应该和dout相同

    Returns:
    - dx: x的梯度
    """
    dx, x = None, cache
    #############################################################################
    #               任务: 实现 ReLU 反向传播.                                   #
    #############################################################################
    dx = dout
    dx[x <= 0] = 0

    #############################################################################
    #                            结束编码                                       #
    #############################################################################
    return dx


def affine_relu_forward(x, w, b):
    """
     ReLU神经元前向传播

    Inputs:
    - x: 输入到 affine层的数据
    - w, b:  affine层的权重矩阵和偏置向量

    Returns 元组:
    - out:  ReLU的输出结果
    - cache: 前向传播的缓存
    """
    ######################################################################
    #               任务: 实现 ReLU神经元前向传播.                       #
    #        注意：你需要调用affine_forward以及relu_forward函数，        #
    #              并将各自的缓存保存在cache中                           #
    ######################################################################

    h, cache_fc = affine_forward(x, w, b)

    out, cache_relu = relu_forward(h)

    cache = (cache_fc, cache_relu)

    ######################################################################
    #                     结束编码                                       #
    ######################################################################
    return out, cache


def affine_relu_backward(dout, cache):
    """
     ReLU神经元的反向传播

    Input:
    - dout: 上层误差梯度
    - cache: affine缓存，以及relu缓存

    Returns:
    - dx: 输入数据x的梯度
    - dw: 权重矩阵w的梯度
    - db: 偏置向量b的梯度
    """
    #############################################################################
    #               任务: 实现 ReLU神经元反向传播.                              #
    #############################################################################
    cache_fc, cache_relu = cache
    dy = relu_backward(dout, cache_relu)
    dx, dw, db = affine_backward(dy, cache_fc)

    #############################################################################
    #                   结束编码                                               #
    #############################################################################
    return dx, dw, db


def softmax_loss(x, y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    ##这里有小bug，log后面不能为0
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


def sigmoid(x):
    """
    数值稳定版本的sigmoid函数。
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def sigmoid_loss(x, y):
    N = x.shape[0]
    Y = np.zeros(x.shape)
    set_Y(Y, y)
    sig_x = sigmoid(x)
    loss = -((1 - Y) * np.log(1 - sig_x) + Y * np.log(sig_x))
    loss = np.sum(loss)
    loss = loss / N
    dx = (sig_x - Y) / N
    return loss, dx


def negative_sample_loss(x, y, rate):
    mask = np.random.random(x.shape)
    mask[mask > (1 - rate)] = 1
    mask[mask < rate] = 0

    N = x.shape[0]
    Y = np.zeros(x.shape)
    set_Y(Y, y)
    sig_x = sigmoid(x)
    loss = -((1 - Y) * np.log(1 - sig_x) + Y * np.log(sig_x))
    loss = np.sum(loss)
    loss = loss / N
    dx = (sig_x - Y) * mask / N
    return loss, dx


def set_Y(Y, y):
    row = Y.shape[0]
    for i in range(row):
        c = y[i]
        Y[i][c] = 1