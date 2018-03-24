from models.net import Net
import tensorflow as tf
from tensorflow.contrib import slim


class PyramidNet(Net):

    def __init__(self, data_provider, model_type, depth, dataset,
                 alpha, weight_decay, nesterov_momentum,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 **kwargs):
        super().__init__(data_provider, model_type, depth, dataset,
                         should_save_logs, should_save_model,
                         renew_logs,
                         **kwargs)
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum

    @property
    def model_identifier(self):
        return "%s_alpha=%s_ds=%s" % (self.model_type, self.alpha, self.dataset_name)

    def _inference(self):
        depth = self.depth
        alpha = self.alpha

        n = (self.depth - 2) / 6
        self.iChannels = 16
        startChannel = 16
        Channeltemp = 16
        addChannel = alpha / (3 * n)

        with tf.variable_scope("first_conv"):
            net = slim.conv2d(self.images, startChannel, [3, 3])
            net = slim.batch_norm(net)

        Channeltemp = startChannel
        startChannel = startChannel + addChannel

    def _configration(self):
        pass

    def _get_optimizer(self):
        return tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)

    def bottleneck(self, _input, output_features, stride):
        n = _input.get_shape()[-1]
        iChannels = n * 4

