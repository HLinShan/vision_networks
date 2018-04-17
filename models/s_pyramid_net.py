import tensorflow as tf
from tensorflow.contrib import slim
from models.pyramid_net import PyramidNet


class SPyramidNet(PyramidNet):
    def __init__(self, data_provider, model_type, depth, dataset,
                 alpha, weight_decay, nesterov_momentum,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 **kwargs):
        super().__init__(data_provider, model_type, depth, dataset,
                         alpha, weight_decay, nesterov_momentum,
                         should_save_logs, should_save_model,
                         renew_logs,
                         **kwargs)

    def basic_block(self, _input, out_channels, stride, c=0):
        with tf.variable_scope("basic_block_%d" % c):
            in_channels = _input.get_shape().as_list()[-1]

            net = slim.batch_norm(_input)
            net = slim.separable_conv2d(net, out_channels, [3, 3], depth_multiplier=1, stride=stride)
            # net = slim.conv2d(net, out_channels, [3, 3], stride=stride)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)
            net = slim.separable_conv2d(net, out_channels, [3, 3], depth_multiplier=1)
            net = slim.batch_norm(net)

            if stride == 2:
                shortcut = slim.avg_pool2d(_input, [2, 2])
            else:
                shortcut = _input
            shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, out_channels - in_channels]])
            return net + shortcut

    def bottleneck(self, _input, out_channels, stride, c=0):
        with tf.variable_scope("bottleneck_%d" % c):
            in_channels = _input.get_shape().as_list()[-1]
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_channels, [1, 1])
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)
            net = slim.separable_conv2d(net, out_channels, [3, 3], stride=stride, depth_multiplier=1)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)
            net = slim.conv2d(net, out_channels * 4, [1, 1])
            net = slim.batch_norm(net)

            if stride == 2:
                shortcut = slim.avg_pool2d(_input, [2, 2])
            else:
                shortcut = _input
            shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, out_channels * 4 - in_channels]])

            return net + shortcut
