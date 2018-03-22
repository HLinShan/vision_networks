from models.my_dense_net import MyDenseNet
import tensorflow as tf
from tensorflow.contrib import slim


class XDenseNet(MyDenseNet):
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 cardinality=2,
                 **kwargs):
        super().__init__(data_provider, growth_rate, depth,
                         total_blocks, keep_prob,
                         weight_decay, nesterov_momentum, model_type, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         reduction,
                         bc_mode,
                         **kwargs)
        self.cardinality = cardinality

    def _inference(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first conv
        with slim.arg_scope(self.arg_scope()):
            with tf.variable_scope("Initial_convolution"):
                net = slim.conv2d(self.images, self.first_output_features, [3, 3])

            for block in range(self.total_blocks):
                with tf.variable_scope("Block_%d" % block):
                    net = self.add_block(net, growth_rate, layers_per_block)

                if block != self.total_blocks - 1:
                    with tf.variable_scope("Transition_after_block_%d" % block):
                        net = self.transition_layer(net)

            with tf.variable_scope("Transition_to_classes"):
                net = slim.batch_norm(net)
                net = tf.reduce_mean(net, axis=[1, 2])
                net = slim.flatten(net)
                logits = slim.fully_connected(net, self.n_classes)

            return logits

    def add_block(self, _input, growth_rate, layers_per_block):
        cardinality = self.cardinality
        nets = []
        for c in range(cardinality):
            nets.append(_input)
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                con_net = tf.concat(nets, axis=3)
                for c in range(cardinality):
                    if layer == 0:
                        net = nets[c]
                    else:
                        net = con_net
                    if self.bc_mode:
                        net = self.bottleneck(net, self.growth_rate * 4, c)
                    net = self.composite_function(net, growth_rate, 3, c)
                    nets[c] = tf.concat([net, nets[c]], axis=3)

        net = tf.concat(nets, axis=3)
        return net
