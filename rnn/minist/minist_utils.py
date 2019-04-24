import os
import struct
import numpy as np
import scipy.misc

bacth_size = 500
feature_num = 784
y_num = 10


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    data = {}
    data['images'] = images
    data['labels'] = labels

    return data


def sample_mnist_minibatch(images, labels, batch_size=100):
    mask = np.random.choice(a=60000, size=batch_size, replace=False, p=None)
    features_tmp = images[mask]
    label = labels[mask]

    features = features_tmp.reshape([batch_size, 28, 28])

    return features, label

# data = load_mnist('/Users/mac/huangjianyi/深度学习/深度学习实战范例/深度学习实战范例/DLAction/MNIST_data', kind='train')
#
# f,l=sample_mnist_minibatch(data['images'],data['labels'])
# scipy.misc.toimage(f[0], cmin=0.0, cmax=1.0).save('./1.jpg')
# scipy.misc.toimage(f[1], cmin=0.0, cmax=1.0).save('./2.jpg')
# print(l[0])
# print(l[1])