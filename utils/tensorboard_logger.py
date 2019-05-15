import os

import numpy as np
import scipy.misc
import tensorflow as tf

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO as StringIO  # Python 3.x


def kernel_to_image(data, padsize=1):
    """Turns a convolutional kernel into an image of nicely tiled filters.
    :param data: numpy array in format N x C x H x W.
    :param padsize: optional int to indicate visual padding between the filters.
    :return: image of the filters in a tiled/mosaic layout
    """
    if len(data.shape) > 4:
        data = np.squeeze(data)
    data = np.transpose(data, (0, 2, 3, 1))
    data_shape = tuple(data.shape)
    min_val = np.min(np.reshape(data, (data_shape[0], -1)), axis=1)
    data = np.transpose((np.transpose(data, (1, 2, 3, 0)) - min_val), (3, 0, 1, 2))
    max_val = np.max(np.reshape(data, (data_shape[0], -1)), axis=1)
    data = np.transpose((np.transpose(data, (1, 2, 3, 0)) / max_val), (3, 0, 1, 2))

    n = int(np.ceil(np.sqrt(data_shape[0])))
    ndim = len(data.shape)
    padding = ((0, n ** 2 - data_shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (ndim - 3)
    data = np.pad(data, padding, mode="constant", constant_values=0)
    # tile the filters into an image
    data_shape = data.shape
    data = np.transpose(np.reshape(data, ((n, n) + data_shape[1:])), ((0, 2, 1, 3) + tuple(range(4, ndim + 1))))
    data_shape = data.shape
    data = np.reshape(data, ((n * data_shape[1], n * data_shape[3]) + data_shape[4:]))
    return (data * 255).astype(np.uint8)


class SummaryWriter(tf.summary.FileWriter):
    def __init__(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        super(SummaryWriter, self).__init__(path)
        import threading

        self.lock = threading.Lock()
        self.count = 0

    def add_summary(self, summary, global_step=None, increment_step_counter=True):
        self.lock.acquire()
        if global_step is None:
            global_step = 0
        if self.count < global_step:
            self.count = global_step
        elif increment_step_counter:
            self.count += 1
        super(SummaryWriter, self).add_summary(summary, self.count)
        self.flush()
        self.lock.release()

    def increment(self):
        self.lock.acquire()
        self.count += 1
        self.lock.release()


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    @property
    def count(self):
        return self.writer.count

    @count.setter
    def count(self, new_count):
        if self.writer.count < new_count:
            self.writer.count = new_count

    def multi_scalar_log(self, tags, values, step):
        for tag, value in zip(tags, values):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step, False)
        self.writer.increment()

    def dict_log(self, items_to_log, step):
        tags, values = zip(*items_to_log.items())
        self.multi_scalar_log(tags, values, step)

    def scalar_summary(self, tag, value, step, increment_counter):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step, increment_counter)

    def network_conv_summary(self, network, step):
        for ii, (name, val) in enumerate(network.state_dict().items()):
            val = val.detach().cpu().numpy()
            name = "layer_%03d/" % ii + name
            if len(val.squeeze().shape) == 4:
                self.conv_variable_summaries(val, step, name, False)
            else:
                self.variable_summaries(val, step, name, False)
        self.writer.increment()

    def network_variable_summary(self, network, step):
        for ii, (name, val) in enumerate(network.state_dict().items()):
            name = "layer_%03d/" % ii + name
            val = val.detach().cpu().numpy()
            self.variable_summaries(val, step, name, False)
        self.writer.increment()

    def variable_summaries(self, var, step, scope="", increment_counter=True):
        # Some useful stats for variables.
        if len(scope) > 0:
            scope = "/" + scope
        scope = "summaries" + scope
        mean = np.mean(np.abs(var))
        self.scalar_summary(scope + "/mean_abs", mean, step, increment_counter)

    def conv_variable_summaries(self, var, step, scope="", increment_counter=True):
        # Useful stats for variables and the kernel images.
        self.variable_summaries(var, step, scope, increment_counter)
        if len(scope) > 0:
            scope = "/" + scope
        scope = "conv_summaries" + scope + "/filters"
        var_shape = var.shape
        if not (var_shape[0] == 1 and var_shape[1] == 1):
            if var_shape[2] < 3:
                var = np.tile(var, [1, 1, 3, 1])
                var_shape = var.shape
            summary_image = kernel_to_image(var[:, :3, :, :])[np.newaxis, ...]
            self.image_summary(scope, summary_image, step, increment_counter)

    def image_summary(self, tag, images, step, increment_counter):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag="%s/%d" % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step, increment_counter)

    def histo_summary(self, tag, values, step, bins=1000, increment_counter=True):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step, increment_counter)
        self.writer.flush()
