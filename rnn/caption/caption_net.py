# -*- coding: utf-8 -*-
import numpy as np
from rnn.baisc.rnn_layer import *
from dnn.basic.layers import *
from rnn.caption.coco_utils import *
from rnn.caption.captioning_trainer import *
import matplotlib.pyplot as plt
#
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#



class CaptioningRNN(object):
    """
    处理图片说明任务RNN网络
    注意：不使用正则化
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn'):
        """
        初始化CaptioningRNN
        Inputs:
        - word_to_idx: 单词字典，用于查询单词索引对应的词向量
        - input_dim: 输入图片数据维度
        - wordvec_dim: 词向量维度.
        - hidden_dim: RNN隐藏层维度.
        - cell_type: 细胞类型; 'rnn' 或 'lstm'.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        word_to_idx.keys()

        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # 初始化词向量
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # 初始化 CNN -> 隐藏层参数，用于将图片特征提取到RNN中
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # 初始化RNN参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # 初始化输出层参数
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

    def loss(self, features, captions):
        """
        计算RNN或LSTM的损失值。
        Inputs:
        - features: 输入图片特征(N, D)。
        - captions: 图像文字说明(N, T)。

        Returns 元组:
        - loss: 损失值。
        - grads:梯度。
        """
        # 将文字切分为两段：captions_in除去最后一词用于RNN输入
        # captions_out除去第一个单词，用于RNN输出配对
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # 掩码
        mask = (captions_out != self._null)

        # 图像仿射转换矩阵
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # 词嵌入矩阵
        W_embed = self.params['W_embed']

        # RNN参数
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # 隐藏层输出转化矩阵
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        #            任务：实现CaptioningRNN传播                                   #
        #     (1)使用仿射变换(features,W_proj,b_proj)，                            #
        #           将图片特征输入进隐藏层初始状态h0(N,H)                          #
        #     (2)使用词嵌入层将captions_in中的单词索引转换为词向量(N,T,W)          #
        #     (3)使用RNN或LSTM处理词向量(N,T,H)                                    #
        #     (4)使用时序仿射传播temporal_affine_forward计算各单词得分(N,T,V)      #
        #     (5)使用temporal_softmax_loss计算损失值                               #
        ############################################################################
        ##forward
        h0, cache_aff = affine_forward(features, W_proj, b_proj)
        x, cahch_word = word_embedding_forward(captions_in, W_embed)

        # h = None
        # cache_h = None
        if self.cell_type == 'rnn':
            h, cache_h = rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            h, cache_h = lstm_forward(x, h0, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)

        out, cahch_out = temporal_affine_forward(h, W_vocab, b_vocab)

        ##loss
        loss, dy = temporal_softmax_loss(out, captions_out, mask, verbose=False)

        ##backward
        dx_ta, dW_vocab, db_vocab = temporal_affine_backward(dy, cahch_out)

        if self.cell_type == 'rnn':
            dx_rnn, dh0, dWx, dWh, db = rnn_backward(dx_ta, cache_h)
        elif self.cell_type == 'lstm':
            dx_rnn, dh0, dWx, dWh, db = lstm_backward(dx_ta, cache_h)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)

        # dx_rnn, dh0, dWx, dWh, db = rnn_backward(dx_ta, cache_h)
        dW_embed = word_embedding_backward(dx_rnn, cahch_word)

        dFeatures, dW_proj, b_proj = affine_backward(dh0, cache_aff)

        ##计算梯度
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db
        grads['W_embed'] = dW_embed
        grads['W_proj'] = dW_proj
        grads['b_proj'] = b_proj

        ############################################################################
        #                             结束编码                                     #
        ############################################################################

        return loss, grads


small_data2 = load_coco_data(max_train=5000)
good_lstm_model = CaptioningRNN(
    cell_type='lstm',
    word_to_idx=small_data2['word_to_idx'],
    input_dim=small_data2['train_features'].shape[1],
    hidden_dim=200,
    wordvec_dim=256 )

good_lstm_solver = CaptioningTrainer(good_lstm_model, small_data2,
                                     update_rule='rmsprop',
                                     num_epochs=50,
                                     batch_size=100,
                                     updater_config={
                                         'learning_rate': 5e-3,
                                     },
                                     lr_decay=0.995,
                                     verbose=True, print_every=50,
                                     )

good_lstm_solver.train()

plt.plot(good_lstm_solver.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()