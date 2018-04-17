import tensorflow as tf
from tensorflow.contrib import slim
from models.net import Net


class Xception(Net):
    def __init__(self, data_provider, model_type, depth, dataset,
                 should_save_logs, should_save_model,
                 keep_prob, weight_decay, nesterov_momentum,
                 renew_logs=False,
                 **kwargs):
        super().__init__(data_provider, model_type, depth, dataset,
                         should_save_logs, should_save_model,
                         renew_logs=renew_logs,
                         **kwargs)
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    def _configration(self):
        print("Xception %d layers"% self.depth)
        pass

    def _inference(self):
        with slim.arg_scope(self.arg_scope()):
            # ===========ENTRY FLOW==============
            with tf.variable_scope("ENTRY_FLOW"):
                # Block 1
                net = slim.conv2d(self.images, 32, [3, 3], padding='same')
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.conv2d(net, 32, [3, 3], padding='same')
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                residual = slim.conv2d(net, 64, [1, 1], stride=2)
                residual = slim.batch_norm(residual,)

                # Block 2
                net = slim.separable_conv2d(net, 64, [3, 3])
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 64, [3, 3])
                net = slim.batch_norm(net)
                net = slim.max_pool2d(net, [3, 3], stride=2, padding="same")
                net = tf.add(net, residual)

            # residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block2_res_conv')
            # residual = slim.batch_norm(residual, scope='block2_res_bn')

            #
            # # Block 3
            # net = tf.nn.relu(net, name='block3_relu1')
            # net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv1')
            # net = slim.batch_norm(net, scope='block3_bn1')
            # net = tf.nn.relu(net, name='block3_relu2')
            # net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv2')
            # net = slim.batch_norm(net, scope='block3_bn2')
            # net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')
            # net = tf.add(net, residual, name='block3_add')
            # residual = slim.conv2d(net, 728, [1, 1], stride=2, scope='block3_res_conv')
            # residual = slim.batch_norm(residual, scope='block3_res_bn')
            #
            # # Block 4
            # net = tf.nn.relu(net, name='block4_relu1')
            # net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv1')
            # net = slim.batch_norm(net, scope='block4_bn1')
            # net = tf.nn.relu(net, name='block4_relu2')
            # net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv2')
            # net = slim.batch_norm(net, scope='block4_bn2')
            # net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block4_max_pool')
            # net = tf.add(net, residual, name='block4_add')

            # ===========MIDDLE FLOW===============
            n = (self.depth-10)//3
            with tf.name_scope("MIDDLE_FLOW"):
                for i in range(n):
                    residual = net
                    net = tf.nn.relu(net)
                    net = slim.separable_conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)
                    net = slim.separable_conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)
                    net = slim.separable_conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = tf.add(net, residual)

            # ========EXIT FLOW============
            with tf.variable_scope("EXIT_FLOW"):
                residual = slim.conv2d(net, 128, [1, 1], stride=2)
                residual = slim.batch_norm(residual)

                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)

                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)

                net = slim.max_pool2d(net, [3, 3], stride=2, padding='same')
                net = tf.add(net, residual)

                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.separable_conv2d(net, 128, [3, 3])
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)

                net = tf.reduce_mean(net, axis=[1, 2])

            logits = slim.fully_connected(net, self.n_classes)

        return logits

    @property
    def model_identifier(self):
        return "%s-ds=%s" % (self.model_type, self.dataset_name)

    def _get_loss(self):
        return NotImplementedError

    def arg_scope(self):
        with slim.arg_scope([slim.batch_norm],
                            scale=True,
                            is_training=self.is_training,
                            updates_collections=None,
                            activation_fn=None):
            with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1):
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
