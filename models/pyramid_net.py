from models.net import Net
import tensorflow as tf
from tensorflow.contrib import slim
import math


class PyramidNet(Net):

    def __init__(self, data_provider, model_type, depth, dataset,
                 alpha, weight_decay, nesterov_momentum,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 **kwargs):
        super().__init__(data_provider, model_type, depth, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         **kwargs)
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    @property
    def model_identifier(self):
        return "%s-%d_alpha=%s_ds=%s" % (self.model_type, self.depth, self.alpha, self.dataset_name)

    def _inference(self):
        # bottleneck
        # n = (self.depth - 2) // 9
        # basic block
        n = (self.depth - 2) // 6
        start_channel = 16
        add_channel = self.alpha / (3 * n)
        layers = []
        with slim.arg_scope(self.arg_scope()):
            with tf.variable_scope("first_conv"):
                # net = slim.batch_norm(net)
                net = slim.conv2d(self.images, start_channel, [3, 3])
                layers.append(net)

            with tf.variable_scope("block1"):
                for i in range(n):
                    start_channel = start_channel + add_channel
                    net = self.basic_block(layers[-1], self.round(start_channel), stride=1, c=i)
                    layers.append(net)

            with tf.variable_scope("block2"):
                for i in range(n):
                    if i == 0:
                        stride = 2
                    else:
                        stride = 1
                    start_channel = start_channel + add_channel
                    net = self.basic_block(layers[-1], self.round(start_channel), stride=stride, c=i)
                    layers.append(net)

            with tf.variable_scope("block3"):
                for i in range(n):
                    if i == 0:
                        stride = 2
                    else:
                        stride = 1
                    start_channel = start_channel + add_channel
                    net = self.basic_block(layers[-1], self.round(start_channel), stride=stride, c=i)
                    layers.append(net)

            with tf.variable_scope("fully_connected"):
                net = slim.batch_norm(layers[-1])
                net = tf.nn.relu(net)
                print(net.get_shape())
                net = tf.reduce_mean(net, axis=[1, 2])
                logits = slim.fully_connected(net, self.n_classes)

        return logits

    def _configration(self):
        pass

    def _get_optimizer(self):
        return tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)

    def basic_block(self, _input, out_channels, stride, c=0):
        with tf.variable_scope("basic_block_%d" % c):
            in_channels = _input.get_shape().as_list()[-1]

            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_channels, [3, 3], stride=stride)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)
            net = slim.conv2d(net, out_channels, [3, 3])
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
            net = slim.conv2d(net, out_channels, [3, 3], stride=stride)
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

    def round(self, x):
        return math.floor(x + 0.5)

    def arg_scope(self):
        with slim.arg_scope([slim.batch_norm],
                            scale=True,
                            is_training=self.is_training,
                            updates_collections=None):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                activation_fn=None,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)
                                ):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=None,
                                    weights_regularizer=slim.l2_regularizer(self.weight_decay)
                                    ) as arg:
                    return arg
