import numpy as np
from rnn.baisc.rnn_layer import *
from dnn.basic.layers import *
from rnn.minist.minist_utils import *
from rnn.minist.minist_trainer import *
import matplotlib.pyplot as plt


class MinstRNN(object):
    """
    处理图片说明任务RNN网络
    注意：不使用正则化
    """

    def __init__(self, input_dim=100, wordvec_dim=28,
                 hidden_dim=128, cell_type='rnn', class_num=10):
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

        # 初始化RNN参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params = {}
        self.cell_type = cell_type
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)
        #        self.params['h0'] = np.zeros([100, dim_mul * hidden_dim])

        # 初始化输出层参数
        self.params['W_vocab'] = np.random.randn(hidden_dim, class_num)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(class_num)

    def loss(self, features, Y):

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
        # captions_in = captions[:, :-1]
        # captions_out = captions[:, 1:]
        [N, H, D] = features.shape

        # RNN参数
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        #
        [_, HH] = Wx.shape

        # 掩码
        h0 = np.zeros([N, HH])

        # 隐藏层输出转化矩阵
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}

        #####前向传播
        if self.cell_type == 'rnn':
            h, cache_h = rnn_forward(features, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            h, cache_h = lstm_forward(features, h0, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)

        # 最后一个隐含层
        last_h = h[:, -1, :]

        out, cahch_out = affine_forward(last_h, W_vocab, b_vocab)

        ##loss
        loss, dout = softmax_loss(out, Y)

        #####后向传播
        dlast_h, dW_vocab, db_vocab = affine_backward(dout, cahch_out)

        dh = np.zeros_like(h)
        dh[:, -1, :] = dlast_h

        if self.cell_type == 'rnn':
            dx_rnn, dh0, dWx, dWh, db = rnn_backward(dh, cache_h)
        elif self.cell_type == 'lstm':
            dx_rnn, dh0, dWx, dWh, db = lstm_backward(dh, cache_h)

        ##计算梯度
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db
        grads['h0'] = dh0

        return loss, grads


data = load_mnist('/Users/mac/huangjianyi/深度学习/深度学习实战范例/深度学习实战范例/DLAction/MNIST_data', kind='train')

good_lstm_model = MinstRNN(
    cell_type='rnn',
    input_dim=100,
    hidden_dim=200,
    wordvec_dim=28)

good_lstm_solver = MinistTrainer(good_lstm_model, data,
                                 update_rule='sgd',
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
