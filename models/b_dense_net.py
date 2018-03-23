from models.my_dense_net import MyDenseNet
import tensorflow as tf
from tensorflow.contrib import slim


class ADenseNet(MyDenseNet):
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
        self.nets = []

    def _inference(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        for i in range(self.layers_per_block):
            self.nets.append(None)
        # first conv
        with slim.arg_scope(self.arg_scope()):
            with tf.variable_scope("Initial_convolution"):
                net = slim.conv2d(self.images, self.first_output_features, [3, 3])

            for block in range(self.total_blocks):
                with tf.variable_scope("Block_%d" % block):
                    for layer in range(layers_per_block):
                        net = self.add_internal_layer(net, growth_rate, layer)
                if block != self.total_blocks - 1:
                    with tf.variable_scope("Transition_after_block_%d" % block):
                        net = self.transition_layer(net)
                        self.nets.append(tf.reduce_mean(net, axis=[1, 2]))

            with tf.variable_scope("Transition_to_classes"):
                self.nets.append(tf.reduce_mean(net, axis=[1, 2]))
                net = tf.concat([self.nets], axis=1)
                net = slim.batch_norm(net)
                net = tf.reduce_mean(net, axis=[1, 2])
                # net = slim.flatten(net)
                logits = slim.fully_connected(net, self.n_classes)

            return logits

    def add_internal_layer(self, _input, growth_rate, layer=0):
        with tf.variable_scope("layer_%d" % layer):
            net = _input
            if self.bc_mode:
                net = self.bottleneck(net, self.growth_rate * 4)

            net = self.composite_function(net, self.growth_rate, 3)
            output = tf.concat([_input, net], axis=3)
        return output
