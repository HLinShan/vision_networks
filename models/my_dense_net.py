from models.net import Net
import tensorflow as tf
from tensorflow.contrib import slim


class MyDenseNet(Net):
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        super().__init__(data_provider, model_type, depth, dataset,
                         should_save_logs, should_save_model,
                         renew_logs)

        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    def _inference(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first conv
        with slim.arg_scope(self.arg_scope()):
            with tf.variable_scope("Conv"):
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

    def _configration(self):
        if not self.bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      self.model_type, self.total_blocks, self.layers_per_block))
        if self.bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      self.model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

    def _get_optimizer(self):
        return tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)

    @property
    def model_identifier(self):
        return "{}_k={}_depth={}_ds={}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

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
            net = self.bottleneck(net, self.growth_rate * 4)
        net = self.composite_function(net, self.growth_rate, 3)

        output = tf.concat(axis=3, values=(_input, net))
        return output

    def bottleneck(self, _input, out_features, i=0):
        with tf.variable_scope("bottleneck_%d" % i):
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_features, [1, 1])
            net = self.dropout(net)
        return net

    def composite_function(self, _input, out_features, kernel_size=3, i=0):
        with tf.variable_scope("composite_function_%d" % i):
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, out_features, [kernel_size, kernel_size])
            net = self.dropout(net)
        return net

    def squeeze_excitation_layer(self, input_x, out_dim, ratio=16, c=0):
        with tf.name_scope('se_%d' % c):
            # out_dim = input_x.get_shape()[-1]
            squeeze = tf.reduce_mean(input_x, axis=[1, 2])
            excitation = slim.fully_connected(squeeze, out_dim // ratio, activation_fn=tf.nn.relu)
            excitation = slim.fully_connected(excitation, out_dim, activation_fn=tf.nn.sigmoid)
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation
            return scale

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

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
