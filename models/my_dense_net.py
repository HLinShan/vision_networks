from models.dense_net import DenseNet
import tensorflow as tf
from tensorflow.contrib import slim


class MyDenseNet(DenseNet):
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        """

        :param data_provider:
        :param growth_rate:
        :param depth:
        :param total_blocks:
        :param keep_prob:
        :param weight_decay:
        :param nesterov_momentum:
        :param model_type:
        :param dataset:
        :param should_save_logs:
        :param should_save_model:
        :param renew_logs:
        :param reduction:
        :param bc_mode:
        :param kwargs:
        """
        super().__init__(data_provider, growth_rate, depth,
                         total_blocks, keep_prob,
                         weight_decay, nesterov_momentum, model_type, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         reduction,
                         bc_mode,
                         **kwargs)

    def _build_graph(self):
        logits = self._inference()
        self.cross_entropy = tf.losses.softmax_cross_entropy(self.labels, logits)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(total_loss)

        prediction = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                net = slim.flatten(net)
                logits = slim.fully_connected(net, self.n_classes)

            return logits

    def transition_layer(self, _input):
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        net = self.composite_function(_input, out_features, kernel_size=1)
        # net = self.squeeze_excitation_layer(net, out_features, 16)
        # run average pooling
        net = slim.avg_pool2d(net, [2, 2])
        return net

    def add_internal_layer(self, _input, growth_rate):
        net = _input
        if self.bc_mode:
            net = self.bottleneck(net, self.growth_rate * 4)

        net = self.composite_function(net, self.growth_rate, 3)

        # concatenate _input with out from composite function
        output = tf.concat(axis=3, values=(_input, net))
        return output

    def bottleneck(self, _input, out_features, i=0):
        with tf.variable_scope("bottleneck_%d" % i):
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_features, [1, 1])
            # net = self.squeeze_excitation_layer(net, out_features, 16)
            net = self.dropout(net)
        return net

    def composite_function(self, _input, out_features, kernel_size=3, i=0):
        with tf.variable_scope("composite_function_%d" % i):
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_features, [kernel_size, kernel_size])
            # net = self.squeeze_excitation_layer(net, out_features, 16)
            net = self.dropout(net)
            # net = self.squeeze_excitation_layer(net, out_features, 16)
        return net

    # def squeeze_excitation_layer(self, _input, out_dim, ratio):
    #     # with tf.name_scope("squeeze_excitation_%i" % i):
    #     squeeze = tf.reduce_mean(_input, axis=[1, 2])
    #     excitation = slim.fully_connected(squeeze, out_dim // ratio)
    #     excitation = slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.relu)
    #     excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    #     scale = _input * excitation
    #
    #     return scale

    def arg_scope(self):
        with slim.arg_scope([slim.batch_norm],
                            scale=True,
                            is_training=self.is_training,
                            updates_collections=None,
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, slim.separable_conv2d],
                                activation_fn=None,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)
                                ):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=None,
                                    weights_regularizer=slim.l2_regularizer(self.weight_decay)
                                    ) as arg:
                    return arg
