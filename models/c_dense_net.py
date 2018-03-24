from models.my_dense_net import MyDenseNet
import tensorflow as tf
from tensorflow.contrib import slim

"""
    多个bottleneck layers
"""

class CDenseNet(MyDenseNet):
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 bc_count=3,
                 **kwargs):

        super().__init__(data_provider, growth_rate, depth,
                         total_blocks, keep_prob,
                         weight_decay, nesterov_momentum, model_type, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         reduction,
                         bc_mode,
                         **kwargs)
        self.bc_count = bc_count

    def _build_graph(self):
        logits = self._inference()
        self.cross_entropy = tf.losses.softmax_cross_entropy(self.labels, logits)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(total_loss)

        # prediction = tf.nn.softmax(logits)
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _configration(self):
        if not self.bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      self.model_type, self.total_blocks, self.layers_per_block))
        if self.bc_mode:
            self.layers_per_block = self.layers_per_block // (self.bc_count + 1)
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      self.model_type, self.total_blocks, self.layers_per_block * self.bc_count,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

    def _inference(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first conv
        with slim.arg_scope(self.arg_scope()):
            with tf.variable_scope("Initial_convolution"):
                net = slim.conv2d(self.images, self.first_output_features, [3, 3])

            for block in range(self.total_blocks):
                with tf.variable_scope("Block_%d" % block):
                    for layer in range(layers_per_block):
                        with tf.variable_scope("layer_%d" % layer):
                            net = self.add_internal_layer(net, growth_rate)

                if block != self.total_blocks - 1:
                    with tf.variable_scope("Transition_after_block_%d" % block):
                        net = self.transition_layer(net)

            with tf.variable_scope("Transition_to_classes"):
                net = slim.batch_norm(net)
                net = tf.reduce_mean(net, axis=[1, 2])
                # net = slim.flatten(net)
                logits = slim.fully_connected(net, self.n_classes)

            return logits

    def transition_layer(self, _input):
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        net = self.composite_function(_input, out_features, kernel_size=1)
        # run average pooling
        net = slim.avg_pool2d(net, [2, 2])
        return net

    def add_internal_layer(self, _input, growth_rate):
        net = _input
        if self.bc_mode:
            for i in range(self.bc_count):
                net = self.bottleneck(net, self.growth_rate * (i + 1), i)
        net = self.composite_function(net, self.growth_rate, 3)

        output = tf.concat(axis=3, values=(_input, net))
        return output

    def bottleneck(self, _input, out_features, i=0):
        with tf.variable_scope("bottleneck_%d" % i):
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_features, [1, 1])
            net = self.dropout(net)
        return net
