# -*- coding: utf-8 -*-
import numpy as np
from utils.gradient_check import *


def rel_error(x, y):
    """相对误差"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    RNN单步前向传播，使用tanh激活单元
    Inputs:
    - x: 当前时间步数据输入(N, D).
    - prev_h: 前一时间步隐藏层状态 (N, H)
    - Wx: 输入层到隐藏层连接权重(D, H)
    - Wh:隐藏层到隐藏层连接权重(H, H)
    - b: 隐藏层偏置项(H,)

    Returns 元组:
    - next_h: 下一隐藏层状态(N, H)
    - cache: 缓存
    """
    next_h, cache = None, None
    ##############################################################################
    #                 任务：实现RNN单步前向传播                                  #
    #               将输出值储存在next_h中，                                     #
    #         将反向传播时所需的各项缓存存放在cache中                            #
    ##############################################################################

    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h = np.tanh(a)
    cache = (x, prev_h, Wx, Wh, b, next_h)

    ##############################################################################
    #                             结束编码                                       #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    RNN单步反向传播。
    Inputs:
    - dnext_h: 后一时间片段的梯度。
    - cache: 前向传播时的缓存。

    Returns 元组:
    - dx: 数据梯度(N, D)。
    - dprev_h: 前一时间片段梯度(N, H)。
    - dWx: 输入层到隐藏层权重梯度(D,H)。
    - dWh:  隐藏层到隐藏层权重梯度(H, H)。
    - db: 偏置项梯度(H,)。
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    #              任务：实现RNN单步反向传播                                     #
    #      提示：tanh(x)梯度:  1 - tanh(x)*tanh(x)                               #
    ##############################################################################
    (x, prev_h, Wh, Wx, b, next_h) = cache
    # (N, H)
    dscores = dnext_h * (1 - next_h * next_h)
    dx = np.dot(dscores, Wx.T)
    dWx = np.dot(x.T, dscores)
    dprev_h = np.dot(dscores, Wh.T)
    dWh = np.dot(prev_h.T, prev_h)
    db = np.sum(dscores, axis=0)

    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    RNN前向传播。
    Inputs:
    - x: 完整的时序数据 (N, T, D)。
    - h0: 隐藏层初始化状态 (N, H)。
    - Wx: 输入层到隐藏层权重 (D, H)。
    - Wh:  隐藏层到隐藏层权重(H, H)。
    - b: 偏置项(H,)。
    Returns 元组:
    - h: 所有时间步隐藏层状态(N, T, H)。
    - cache: 反向传播所需的缓存。
    """
    h, cache = None, None
    ##############################################################################
    #                     任务：实现RNN前向传播。                                #
    #        提示： 使用前面实现的rnn_step_forward 函数。                        #
    ##############################################################################

    N, T, D = x.shape
    _, H = h0.shape

    h = np.zeros((N, T, H))
    prev_h = h0

    for i in range(T):
        x_input = x[:, i, :]
        next_h, _ = rnn_step_forward(x_input, prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:, i, :] = prev_h

    cache = (x, h0, Wh, Wx, b, h)

    ##############################################################################
    #                           结束编码                                         #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    RNN反向传播。
    Inputs:
    - dh: 隐藏层所有时间步梯度(N, T, H)。
    Returns 元组:
    - dx: 输入数据时序梯度(N, T, D)。
    - dh0: 初始隐藏层梯度(N, H)。
    - dWx: 输入层到隐藏层权重梯度(D, H)。
    - dWh: 隐藏层到隐藏层权重梯度(H, H)。
    - db: 偏置项梯度(H,)。
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    #                任务：实现RNN反向传播。                                     #
    #            提示：使用 rnn_step_backward函数。                              #
    ##############################################################################
    x, h0, Wh, Wx, b, h = cache
    N, T, H = dh.shape
    _, _, D = x.shape
    next_h = h[:, T - 1, :]
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    for t in range(T):
        t = T - 1 - t
        xt = x[:, t, :]
        if t == 0:
            prev_h = h0
        else:
            prev_h = h[:, t - 1, :]
        step_cache = (xt, prev_h, Wh, Wx, b, next_h)
        next_h = prev_h
        dnext_h = dh[:, t, :] + dprev_h
        dx[:, t, :], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, step_cache)
        dWx, dWh, db = dWx + dWxt, dWh + dWht, db + dbt
    dh0 = dprev_h
    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    词嵌入前向传播，将数据矩阵中的N条长度为T的词索引转化为词向量。
    如：W[x[i,j]]表示第i条，第j时间步单词索引所对应的词向量。
    Inputs:
    - x: 整数型数组(N,T),N表示数据条数，T表示单条数据长度，
      数组的每一元素存放着单词索引，取值范围[0,V)。
    - W: 词向量矩阵(V,D)存放各单词对应的向量。

    Returns 元组:
    - out:输出词向量(N, T, D)。
    - cache:反向传播时所需的缓存。
    """
    out, cache = None, None
    ##############################################################################
    #                     任务：实现词嵌入前向传播。                             #
    ##############################################################################
    N, T = x.shape
    H, D = W.shape

    out = np.zeros((N, T, D))

    for i in range(N):
        for j in range(T):
            out[i, j, :] = W[x[i, j]]

    cache = (x, W.shape)

    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    词嵌入反向传播

    Inputs:
    - dout: 上层梯度 (N, T, D)
    - cache:前向传播缓存

    Returns:
    - dW: 词嵌入矩阵梯度(V, D).
    """
    dW = None
    ##############################################################################
    #                 任务：实现词嵌入反向传播                                   #
    #             提示：你可以使用np.add.at函数                                  #
    #         例如 np.add.at(a,[1,2],1)相当于a[1],a[2]分别加1                    #
    ##############################################################################
    x, W_shape = cache
    dW = np.zeros(W_shape)
    np.add.at(dW, x, dout)

    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################
    return dW


def temporal_affine_forward(x, w, b):
    """
    时序隐藏层仿射传播：将隐藏层时序数据(N,T,D)重塑为(N*T,D)，
    完成前向传播后，再重塑回原型输出。

    Inputs:
    - x: 时序数据(N, T, D)。
    - w: 权重(D, M)。
    - b: 偏置(M,)。

    Returns 元组:
    - out: 输出(N, T, M)。
    - cache: 反向传播缓存。
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    时序隐藏层仿射反向传播。

    Input:
    - dout:上层梯度 (N, T, M)。
    - cache: 前向传播缓存。

    Returns 元组:
    - dx: 输入梯度(N, T, D)。
    - dw: 权重梯度 (D, M)。
    - db: 偏置项梯度 (M,)。
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    时序版本的Softmax损失和原版本类似，只需将数据(N, T, V)重塑为(N*T,V)即可。
    需要注意的是，对于NULL标记不计入损失值，因此，你需要加入掩码进行过滤。
    Inputs:
    - x: 输入数据得分(N, T, V)。
    - y: 目标索引(N, T)，其中0<= y[i, t] < V。
    - mask: 过滤NULL标记的掩码。
    Returns 元组:
    - loss: 损失值。
    - dx: x梯度。
    """
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

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


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    LSTM单步前向传播

    Inputs:
    - x: 输入数据 (N, D)
    - prev_h: 前一隐藏层状态 (N, H)
    - prev_c: 前一细胞状态(N, H)
    - Wx: 输入层到隐藏层权重(D, 4H)
    - Wh: 隐藏层到隐藏层权重 (H, 4H)
    - b: 偏置项(4H,)

    Returns 元组:
    - next_h:  下一隐藏层状态(N, H)
    - next_c:  下一细胞状态(N, H)
    - cache: 反向传播所需的缓存
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    #              任务：实现LSTM单步前向传播。                                 #
    #         提示：稳定版本的sigmoid函数已经帮你实现，直接调用即可。           #
    #               tanh函数使用np.tanh。                                       #
    #############################################################################
    H = prev_h.shape[1]

    # dim:（N,D+H）
    x_c = np.concatenate((prev_h, x), axis=1)
    # dim:（D+H,H）
    w_c = np.concatenate((Wh, Wx), axis=0)
    # forget_gate（N,H）
    input_gate = sigmoid(np.dot(x_c, w_c[:, 0:H]) + b[0:H])
    # input_gate（N,H）
    forget_gate = sigmoid(np.dot(x_c, w_c[:, 1 * H:2 * H]) + b[1 * H:2 * H])
    # dim(N,H)
    output_gate = sigmoid(np.dot(x_c, w_c[:, 2 * H:3 * H]) + b[2 * H:3 * H])
    # dim(N,H)
    Ct = np.tanh(np.dot(x_c, w_c[:, 3 * H:4 * H]) + b[3 * H:4 * H])
    # dim(N,H)
    next_c = forget_gate * prev_c + input_gate * Ct

    next_c_score = np.tanh(next_c)
    next_h = output_gate * next_c_score

    cache = (x, Wx, Wh, b, Ct, input_gate, output_gate,
             forget_gate, prev_h, prev_c, next_c)

    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
     LSTM单步反向传播

    Inputs:
    - dnext_h: 下一隐藏层梯度 (N, H)
    - dnext_c: 下一细胞梯度 (N, H)
    - cache: 前向传播缓存

    Returns 元组:
    - dx: 输入数据梯度 (N, D)
    - dprev_h: 前一隐藏层梯度 (N, H)
    - dprev_c: 前一细胞梯度(N, H)
    - dWx: 输入层到隐藏层梯度(D, 4H)
    - dWh:  隐藏层到隐藏层梯度(H, 4H)
    - db:  偏置梯度(4H,)
    """
    dx, dprev_h, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    #                           任务：实现LSTM单步反向传播                      #
    #          提示：sigmoid(x)函数梯度：sigmoid(x)*(1-sigmoid(x))              #
    #                tanh(x)函数梯度：   1-tanh(x)*tanh(x)                      #
    #############################################################################

    x, Wx, Wh, b, Ct, input_gate, output_gate, forget_gate, prev_h, prev_c, next_c = cache

    [N, D] = x.shape
    H = prev_h.shape[1]

    dx = np.zeros([N, D])
    dprev_h = np.zeros([N, H])
    dWx = np.zeros([D, 4 * H])
    dWh = np.zeros([H, 4 * H])
    db = np.zeros([4 * H])

    dx_c = np.zeros([N, D + H])
    dw_c = np.zeros([D + H, 4 * H])

    # dim:（N,D+H）
    x_c = np.concatenate((prev_h, x), axis=1)
    # dim:（D+H,H）
    w_c = np.concatenate((Wh, Wx), axis=0)

    douput_gate = dnext_h * np.tanh(next_c)
    dnext_c_score = dnext_h * output_gate

    dnext_c = dnext_c_score * (1 - np.tanh(next_c) ** 2) + dnext_c
    dforget_gate = dnext_c * prev_c
    dprev_c = dnext_c * forget_gate
    dinput_gate = dnext_c * Ct
    dCt = dnext_c * input_gate

    x_tmp1 = dCt * (1 - Ct ** 2)
    dx_c_tmp1 = x_tmp1.dot(w_c[:, 3 * H:4 * H].T)
    dw_c[:, 3 * H:4 * H] = x_c.T.dot(x_tmp1)
    db[3 * H:4 * H] = np.sum(x_tmp1, axis=0)

    x_tmp2 = douput_gate * output_gate * (1 - output_gate)
    dx_c_tmp2 = x_tmp2.dot(w_c[:, 2 * H:3 * H].T)
    dw_c[:, 2 * H:3 * H] = x_c.T.dot(x_tmp2)
    db[2 * H:3 * H] = np.sum(x_tmp2, axis=0)

    x_tmp3 = dforget_gate * forget_gate * (1 - forget_gate)
    dx_c_tmp3 = x_tmp3.dot(w_c[:, 1 * H:2 * H].T)
    dw_c[:, 1 * H:2 * H] = x_c.T.dot(x_tmp3)
    db[1 * H:2 * H] = np.sum(dforget_gate * forget_gate * (1 - forget_gate), axis=0)

    x_tmp4 = dinput_gate * input_gate * (1 - input_gate)
    dx_c_tmp4 = x_tmp4.dot(w_c[:, 0: H].T)
    dw_c[:, 0: H] = x_c.T.dot(x_tmp4)
    db[0: H] = np.sum(x_tmp4, axis=0)

    dx_c = dx_c_tmp1 + dx_c_tmp2 + dx_c_tmp3 + dx_c_tmp4

    dprev_h = dx_c[:, 0:H]
    dx = dx_c[:, H:D + H]
    dWh = dw_c[0:H, ]
    dWx = dw_c[H:H + D, ]
    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    LSTM前向传播
    Inputs:
    - x: 输入数据 (N, T, D)
    - h0:初始化隐藏层状态(N, H)
    - Wx: 输入层到隐藏层权重 (D, 4H)
    - Wh: 隐藏层到隐藏层权重(H, 4H)
    - b: 偏置项(4H,)

    Returns 元组:
    - h: 隐藏层所有状态 (N, T, H)
    - cache: 用于反向传播的缓存
    """
    h, cache = None, None
    #############################################################################
    #                    任务： 实现完整的LSTM前向传播                          #
    #############################################################################
    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    prev_c = np.zeros([N, H])
    cache = {}
    h = np.zeros((N, T, H))

    for i in range(T):
        next_h, next_c, cache[i] = lstm_step_forward(x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:, i, :] = prev_h
    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################
    return h, cache


def lstm_backward(dh, cache):
    """
    LSTM反向传播
    Inputs:
    - dh: 各隐藏层梯度(N, T, H)
    - cache: V前向传播缓存

    Returns 元组:
    - dx: 输入数据梯度 (N, T, D)
    - dh0:初始隐藏层梯度(N, H)
    - dWx: 输入层到隐藏层权重梯度 (D, 4H)
    - dWh: 隐藏层到隐藏层权重梯度 (H, 4H)
    - db: 偏置项梯度 (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    #                 任务：实现完整的LSTM反向传播                              #
    #############################################################################
    N, T, H = dh.shape
    x, Wx, Wh, b, input, input_gate, output_gate, forget_gate, prev_h, prev_c, next_scores_c = cache[T - 1]
    D = x.shape[1]
    dnext_h = np.zeros([N, H])
    dnext_c = np.zeros([N, H])
    dx = np.zeros([N, T, D])
    dWx = np.zeros([D, 4 * H])
    dWh = np.zeros([H, 4 * H])
    db = np.zeros([4 * H, ])

    for i in range(T):
        t = T - i - 1
        dnext_h = dnext_h + dh[:, t, :]
        dx[:, t, :], dprev_h, dprev_c, dWx_tmp, dWh_tmp, db_tmp = lstm_step_backward(dnext_h, dnext_c, cache[t])
        dnext_h = dprev_h
        dnext_c = dprev_c

        dWx = dWx + dWx_tmp
        dWh = dWh + dWh_tmp
        db = db + db_tmp

    dh0 = dprev_h

    ##############################################################################
    #                               结束编码                                     #
    ##############################################################################

    return dx, dh0, dWx, dWh, db

# np.random.seed(0)
#
# N, D, T, H = 2, 3, 10, 6
#
# x = np.random.randn(N, T, D)
# h0 = np.random.randn(N, H)
# Wx = np.random.randn(D, 4 * H)
# Wh = np.random.randn(H, 4 * H)
# b = np.random.randn(4 * H)
#
# out, cache = lstm_forward(x, h0, Wx, Wh, b)
#
# dout = np.random.randn(*out.shape)
#
# dx, dh0, dWx, dWh, db = lstm_backward(dout, cache)
#
# fx = lambda x: lstm_forward(x, h0, Wx, Wh, b)[0]
# fh0 = lambda h0: lstm_forward(x, h0, Wx, Wh, b)[0]
# fWx = lambda Wx: lstm_forward(x, h0, Wx, Wh, b)[0]
# fWh = lambda Wh: lstm_forward(x, h0, Wx, Wh, b)[0]
# fb = lambda b: lstm_forward(x, h0, Wx, Wh, b)[0]
#
# dx_num = eval_numerical_gradient_array(fx, x, dout)
# dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
# dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
# dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
# db_num = eval_numerical_gradient_array(fb, b, dout)
#
# print('dx 误差: ', rel_error(dx_num, dx))
# print('dh0 误差: ', rel_error(dx_num, dx))
# print('dWx 误差: ', rel_error(dx_num, dx))
# print('dWh 误差: ', rel_error(dx_num, dx))
# print('db 误差: ', rel_error(dx_num, dx))
