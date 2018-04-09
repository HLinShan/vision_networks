from models.resnet import Resnet
from tensorflow.contrib import slim
import tensorflow as tf


class SResnet(Resnet):

    def _inference(self):
        n = self.residual_blocks_per_block
        layers = []
        with slim.arg_scope(self.arg_scope()):
            # net = slim.batch_norm(self.images)
            net = slim.conv2d(self.images, 16, [3, 3])
            layers.append(net)
            with tf.variable_scope("block1"):
                for i in range(n):
                    with tf.variable_scope("residual_block_%d" % i):
                        net = self.residual_block(layers[-1], 16)
                        layers.append(net)

            # with tf.variable_scope("SE_block1"):
            #     net = self.squeeze_excitation_layer(layers[-1], 16)
            #     net = slim.avg_pool2d(net, [2, 2])
            #     layers.append(net)

            with tf.variable_scope("block2"):
                for i in range(n):
                    with tf.variable_scope("residual_block_%d" % i):
                        net = self.residual_block(layers[-1], 32)
                        layers.append(net)
            #
            # with tf.variable_scope("SE_block2"):
            #     net = self.squeeze_excitation_layer(layers[-1], 32)
            #     net = slim.avg_pool2d(net, [2, 2])
            #     layers.append(net)

            with tf.variable_scope("block3"):
                for i in range(n):
                    with tf.variable_scope("residual_block_%d" % i):
                        net = self.residual_block(layers[-1], 64)
                        layers.append(net)

            # with tf.variable_scope("SE_block3"):
            #     net = self.squeeze_excitation_layer(layers[-1], 64)
            #     net = slim.avg_pool2d(net,[2,2])
            #     layers.append(net)

            with tf.variable_scope("fc"):
                net = slim.batch_norm(layers[-1])
                net = tf.nn.relu(net)
                net = tf.reduce_mean(net, [1, 2])
                logits = slim.fully_connected(net, self.n_classes)
        return logits

    def residual_block(self, _input, num_outputs):
        num_inputs = _input.get_shape().as_list()[-1]
        if num_outputs == num_inputs:
            stride = 1
        else:
            stride = 2

        net = slim.batch_norm(_input)
        net = slim.conv2d(net, num_outputs, [3, 3], stride=stride)

        net = slim.batch_norm(net)
        net = tf.nn.relu(net)
        net = slim.separable_conv2d(net, num_outputs, [3, 3], depth_multiplier=1)
        net = slim.batch_norm(net)

        if stride == 1:
            net = net + _input
        else:
            pad = slim.avg_pool2d(_input, [2, 2])
            net = net + tf.pad(pad, [[0, 0], [0, 0], [0, 0], [0, num_inputs]])

        return net

    @property
    def model_identifier(self):
        return "%s-SE-%d_ds=%s_3" % (self.model_type, self.depth, self.dataset_name)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio=16, c=0):
        with tf.name_scope('SE_Block_%d' % c):
            # out_dim = input_x.get_shape()[-1]
            squeeze = tf.reduce_mean(input_x, axis=[1, 2])
            excitation = slim.fully_connected(squeeze, out_dim // ratio, activation_fn=tf.nn.relu)
            excitation = slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.sigmoid)
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation
            return scale
