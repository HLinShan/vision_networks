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

    # def _inference(self):
    #     growth_rate = self.growth_rate
    #     layers_per_block = self.layers_per_block
    #     # first conv
    #     with slim.arg_scope(self.arg_scope()):
    #         with tf.variable_scope("Initial_convolution"):
    #             net = slim.conv2d(self.images, self.first_output_features, [3, 3])
    #
    #         for block in range(self.total_blocks):
    #             with tf.variable_scope("Block_%d" % block):
    #                 for layer in range(layers_per_block):
    #                     with tf.variable_scope("layer_%d" % layer):
    #                         net = self.add_internal_layer(net, growth_rate)
    #
    #             if block != self.total_blocks - 1:
    #                 with tf.variable_scope("Transition_after_block_%d" % block):
    #                     net = self.transition_layer(net)
    #
    #         with tf.variable_scope("Transition_to_classes"):
    #             net = slim.batch_norm(net)
    #             net = tf.reduce_mean(net, axis=[1, 2])
    #             net = slim.flatten(net)
    #             logits = slim.fully_connected(net, self.n_classes)
    #
    #         return logits

    def add_internal_layer(self, _input, growth_rate):
        net = _input
        net = slim.batch_norm(net)
        net = slim.separable_conv2d(net, self.growth_rate, [3, 3], depth_multiplier=1)
        # concatenate _input with out from composite function
        net = slim.batch_norm(net)
        net = slim.separable_conv2d(net, self.growth_rate, [3, 3], depth_multiplier=1)
        output = tf.concat(axis=3, values=(_input, net))
        return output

    # def bottleneck(self, _input, out_features):
    #     with tf.variable_scope("bottleneck"):
    #         net = slim.batch_norm(_input)
    #         net = slim.conv2d(net, out_features, [1, 1])
    #         net = self.dropout(net)
    #     return net

    # def composite_function(self, _input, out_features, kernel_size=3):
    #     with tf.variable_scope("composite_function"):
    #         net = slim.batch_norm(_input)
    #         net = slim.conv2d(net, out_features, [kernel_size, kernel_size])
    #         net = self.dropout(net)
    #     return net
