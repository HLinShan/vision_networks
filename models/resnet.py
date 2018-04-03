from models.net import Net
from tensorflow.contrib import slim
import tensorflow as tf


class Resnet(Net):

    def __init__(self, data_provider, model_type, depth, dataset,
                 weight_decay, nesterov_momentum, keep_prob,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 **kwargs):
        super().__init__(data_provider, model_type, depth, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         **kwargs)
        self.residual_blocks_per_block = (depth - 2) // 6
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    def _configration(self):
        print("depth %d,residual_blocks_per_block per %d" % (self.depth, self.residual_blocks_per_block))

    def _inference(self):
        n = self.residual_blocks_per_block
        layers = []
        with slim.arg_scope(self.arg_scope()):
            net = slim.batch_norm(self.images)
            net = slim.conv2d(net, 16, [3, 3])
            layers.append(net)
            with tf.variable_scope("block1"):
                for i in range(n):
                    with tf.variable_scope("residual_block_%d" % i):
                        net = self.residual_block(layers[-1], 16)
                        layers.append(net)

            with tf.variable_scope("block2"):
                for i in range(n):
                    with tf.variable_scope("residual_block_%d" % i):
                        net = self.residual_block(layers[-1], 32)
                        layers.append(net)

            with tf.variable_scope("block3"):
                for i in range(n):
                    with tf.variable_scope("residual_block_%d" % i):
                        net = self.residual_block(layers[-1], 64)
                        layers.append(net)

            with tf.variable_scope("fc"):
                net = slim.batch_norm(layers[-1])
                net = tf.reduce_mean(net, [1, 2])
                logits = slim.fully_connected(net, self.n_classes)
        return logits

    @property
    def model_identifier(self):
        return "%s_depth=%d_ds=%s" % (self.model_type, self.depth, self.dataset_name)

    def residual_block(self, _input, num_outputs):
        num_inputs = _input.get_shape().as_list()[-1]
        if num_outputs == num_inputs:
            stride = 1
        else:
            stride = 2

        with tf.variable_scope("conv1"):
            net = slim.batch_norm(_input)
            net = slim.conv2d(net, num_outputs, [3, 3], stride)
        with tf.variable_scope("conv2"):
            net = slim.batch_norm(net)
            net = slim.conv2d(net, num_outputs, [3, 3])

        if stride == 1:
            net = net + _input
        else:
            pad = slim.avg_pool2d(_input, [2, 2])
            net = net + tf.pad(pad, [[0, 0], [0, 0], [0, 0], [num_inputs // 2, num_inputs // 2]])

        return net

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
