from models.my_dense_net import MyDenseNet
import tensorflow as tf
from tensorflow.contrib import slim


class SDenseNet(MyDenseNet):
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        super().__init__(data_provider, growth_rate, depth,
                         total_blocks, keep_prob,
                         weight_decay, nesterov_momentum, model_type, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         reduction,
                         bc_mode,
                         **kwargs)

    def add_internal_layer(self, _input, growth_rate):
        net = _input
        net = slim.batch_norm(net)
        net = slim.conv2d(net, self.growth_rate * 4, [1, 1], depth_multiplier=1)
        net = slim.batch_norm(net)
        net = slim.separable_conv2d(net, self.growth_rate, [3, 3], depth_multiplier=1)
        # net = slim.batch_norm(net, activation_fn=None)
        output = tf.concat(axis=3, values=(_input, net))
        return output

    def transition_layer(self, _input):
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        net = self.composite_function(_input, out_features, kernel_size=1)
        net = self.squeeze_excitation_layer(net, out_features)
        # run average pooling
        net = slim.avg_pool2d(net, [2, 2])

        return net

    def squeeze_excitation_layer(self, input_x, out_dim, ratio=16, c=0):
        with tf.name_scope('SE_Block_%d' % c):
            # out_dim = input_x.get_shape()[-1]
            squeeze = tf.reduce_mean(input_x, axis=[1, 2])
            excitation = slim.fully_connected(squeeze, out_dim // ratio, activation_fn=tf.nn.relu)
            excitation = slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.sigmoid)
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation
            return scale

    @property
    def model_identifier(self):
        return "{}_k={}_depth={}_ds={}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)
