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
        net = slim.separable_conv2d(net, self.growth_rate * 4, [3, 3], depth_multiplier=1)
        net = slim.batch_norm(net)
        net = slim.separable_conv2d(net, self.growth_rate, [3, 3], depth_multiplier=1)
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
