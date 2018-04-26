# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks
(also known as ResNet v2).

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def resnet_v2(
    resnet_size, num_classes, feature_attention, data_format,
    use_tpu, apply_to="input", extra_convs=0):
  return resnet_v2_generator(
    resnet_size=resnet_size, num_classes=num_classes,
    feature_attention=feature_attention, data_format=data_format, use_tpu=use_tpu,
    apply_to=apply_to, extra_convs=extra_convs)


def resnet_v2_generator(
    resnet_size, num_classes, feature_attention, data_format, use_tpu, apply_to, extra_convs):
  """Generator for ResNet v1 models.
  A
rgs:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """
  assert data_format == 'channels_last'
  def model(inputs, is_training):
    drew_resnet = DrewResnet(
                weight_path=None,
                trainable=True,
                apply_to=apply_to,
                extra_convs=extra_convs,
                resnet_size=resnet_size,
                use_tpu=use_tpu)
    logits = drew_resnet.build(
        inputs,
        output_shape=num_classes,
        training=is_training,
        feature_attention=feature_attention)
    return logits
  model.default_image_size = 224
  return model

class DrewResnet:
  """Base class for building the Resnet v2 Model.
  """

  def __init__(
      self,
      weight_path=None,
      trainable=True,
      finetune_layers=None,
      ILSVRC_activation='relu',
      finetune_activation='selu',
      num_classes=1000,
      num_filters=64,
      kernel_size=7,
      conv_stride=2,
      first_pool_size=3,
      first_pool_stride=2,
      second_pool_size=7,
      second_pool_stride=1,
      resnet_size=18,
      block_strides=[1, 2, 2, 2],
      data_format=None,
      apply_to='input',
      extra_convs=1,
      squash=tf.sigmoid,
      output_layer='final_dense',
      human_score_layer='final_avg_pool',
      probability_layer='prob',
      use_batchnorm=False,
      use_tpu=False):
    assert weight_path is None
    self.resnet_size = resnet_size
    block_sizes = _get_block_sizes(resnet_size)
    attention_blocks = _get_attention_sizes(resnet_size)
    data_format = 'channels_last'
    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      block_fn = self.building_block
      final_size = 512
    else:
      block_fn = self.bottleneck_block
      final_size = 2048

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.second_pool_size = second_pool_size
    self.second_pool_stride = second_pool_stride
    self.block_fn = block_fn
    self.block_sizes = block_sizes
    self.attention_blocks = attention_blocks
    self.block_strides = block_strides
    self.final_size = final_size
    self.output_layer = output_layer
    self.probability_layer = probability_layer
    self.use_batchnorm = use_batchnorm
    self.use_tpu = use_tpu
    self.apply_to = apply_to
    self.trainable = trainable
    self.squash = squash
    self.extra_convs = extra_convs
    self.attention_losses = []
    if isinstance(self.squash, str):
      self.squash = interpret_nl(self.squash)

  def __getitem__(self, name):
    return getattr(self, name)

  def __contains__(self, name):
    return hasattr(self, name)

  def build(
      self,
      rgb,
      output_shape=None,
      training=True,
      activation=True,
      feature_attention=False):

    assert rgb.get_shape().as_list()[1:] == [224, 224, 3]

    # rgb = tf.Print(rgb, [tf.shape(rgb)], "tf.shape(rgb)")

    inputs = self.conv2d_fixed_padding(
        inputs=rgb,
        filters=self.num_filters,
        kernel_size=self.kernel_size,
        strides=self.conv_stride,
        data_format=self.data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    if self.first_pool_size:
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

    for i, (num_blocks, use_attention) in enumerate(
            zip(self.block_sizes, self.attention_blocks)):
        num_filters = self.num_filters * (2**i)
        if isinstance(use_attention, list):
            block_attention = [
                feature_attention if x else '' for x in use_attention]
        else:
            block_attention = num_blocks * [
                use_attention * feature_attention]
        assert num_blocks == len(
            block_attention), 'Fix your attention application.'
        inputs = self.block_layer(
            inputs=inputs,
            filters=num_filters,
            block_fn=self.block_fn,
            blocks=num_blocks,
            strides=self.block_strides[i],
            training=training,
            name='block_layer{}'.format(i + 1),
            data_format=self.data_format,
            feature_attention=block_attention)

    inputs = self.batch_norm_relu(inputs, training, self.data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs,
        pool_size=self.second_pool_size,
        strides=self.second_pool_stride,
        padding='VALID',
        data_format=self.data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')

    self.embedding = tf.reshape(inputs, [-1, self.final_size])
    dense_output = tf.layers.dense(inputs=self.embedding, units=self.num_classes)
    final_dense = tf.identity(dense_output, 'final_dense')
    prob = tf.nn.softmax(
        final_dense,
        name='softmax_tensor')
    setattr(self, self.output_layer, final_dense)
    setattr(self, self.probability_layer, prob)
    return dense_output

  def feature_attention(
          self,
          bottom,
          global_pooling=tf.reduce_mean,
          # intermediate_nl=tf.nn.relu,
          intermediate_nl=tf.nn.tanh,
          squash=tf.sigmoid,
          name=None,
          training=True,
          combine='sum_p',
          _BATCH_NORM_DECAY=0.997,
          _BATCH_NORM_EPSILON=1e-5,
          r=4,
          return_map=False):
    """https://arxiv.org/pdf/1709.01507.pdf"""
    # 1. Global pooling
    mu = global_pooling(
        bottom, reduction_indices=[1, 2], keep_dims=True)

    # 2. FC layer with c / r channels + a nonlinearity
    c = int(mu.get_shape()[-1])
    intermediate_size = int(c / r)
    intermediate_activities = intermediate_nl(
        self.fc_layer(
            bottom=tf.contrib.layers.flatten(mu),
            out_size=intermediate_size,
            name='%s_ATTENTION_intermediate' % name,
            training=training))

    # intermediate_activities = tf.Print(
    #   intermediate_activities, [intermediate_activities], "intermediate_activities")

    # 3. FC layer with c / r channels + a nonlinearity
    if squash is not None:
      out_size = c
      output_activities = self.fc_layer(
        bottom=intermediate_activities,
        out_size=out_size,
        name='%s_ATTENTION_output' % name,
        training=training)

    # 4. Add batch_norm to scaled activities
    if self.use_batchnorm:
      bottom = tf.layers.batch_normalization(
        inputs=bottom,
        axis=3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True)

    # 5. Scale bottom with output_activities
    exp_activities = tf.expand_dims(
        tf.expand_dims(output_activities, 1), 1)
    if return_map:
      return exp_activities
    # scaled_bottom = bottom * tf.cast(exp_activities, dtype=bottom.dtype)
    scaled_bottom = bottom * exp_activities

    # 6. Add a loss term to compare scaled activity to clickmaps
    if combine == 'sum_abs':
      salience_bottom = tf.reduce_sum(
            tf.abs(
                scaled_bottom), axis=-1, keep_dims=True)
    elif combine == 'sum_p':
      salience_bottom = tf.reduce_sum(
            tf.pow(
                scaled_bottom, 2), axis=-1, keep_dims=True)
    else:
      raise NotImplementedError(
            '%s combine not implmented.' % combine)
    self.attention_losses += [salience_bottom]
    return scaled_bottom

  def feature_attention_fc(
          self,
          bottom,
          intermediate_nl=tf.nn.tanh,
          squash=tf.sigmoid,
          name=None,
          training=True,
          extra_convs=2,
          extra_conv_size=3,
          dilation_rate=(1, 1),
          intermediate_kernel=1,
          normalize_output=False,
          include_fa=True,
          interaction='both',
          r=4):
    """Fully convolutional form of https://arxiv.org/pdf/1709.01507.pdf"""

    # 1. FC layer with c / r channels + a nonlinearity
    c = int(bottom.get_shape()[-1])
    intermediate_channels = int(c / r)
    intermediate_activities = tf.layers.conv2d(
        inputs=bottom,
        filters=intermediate_channels,
        kernel_size=intermediate_kernel,
        activation=intermediate_nl,
        padding='SAME',
        use_bias=True,
        kernel_initializer=tf.variance_scaling_initializer(),
        trainable=training,
        name='%s_ATTENTION_intermediate' % name)

    # 1a. Optionally add convolutions with spatial dimensions
    if extra_convs:
        for idx in range(extra_convs):
            if self.use_batchnorm:
                intermediate_activities = self.batch_norm_relu(
                    inputs=intermediate_activities,
                    training=training,
                    use_relu=False)
            intermediate_activities = tf.layers.conv2d(
                inputs=intermediate_activities,
                filters=intermediate_channels,
                kernel_size=extra_conv_size,
                activation=intermediate_nl,
                padding='SAME',
                use_bias=True,
                dilation_rate=dilation_rate,
                kernel_initializer=tf.variance_scaling_initializer(),
                trainable=training,
                name='%s_ATTENTION_intermediate_%s' % (name, idx))

    # 2. Spatial attention map
    output_activities = tf.layers.conv2d(
        inputs=intermediate_activities,
        filters=1,  # c,
        kernel_size=1,
        padding='SAME',
        use_bias=True,
        activation=None,
        kernel_initializer=tf.variance_scaling_initializer(),
        trainable=training,
        name='%s_ATTENTION_output' % name)
    if self.use_batchnorm:
        output_activities = self.batch_norm_relu(
            inputs=output_activities,
            training=training,
            use_relu=False)

    # Also calculate se attention
    if include_fa:
        fa_map = self.feature_attention(
            bottom=bottom,
            intermediate_nl=intermediate_nl,
            squash=None,
            name=name,
            training=training,
            r=r,
            return_map=True)
        if interaction == 'both':
            k = fa_map.get_shape().as_list()[-1]
            alpha = tf.get_variable(
                name='alpha_%s' % name,
                shape=[1, 1, 1, k],
                initializer=tf.variance_scaling_initializer())
            beta = tf.get_variable(
                name='beta_%s' % name,
                shape=[1, 1, 1, k],
                initializer=tf.variance_scaling_initializer())
            additive = output_activities + fa_map
            multiplicative = output_activities * fa_map
            output_activities = alpha * additive + beta * multiplicative
            # output_activities = output_activities * fa_map
        elif interaction == 'multiplicative':
            output_activities = output_activities * fa_map
        elif interaction == 'additive':
            output_activities = output_activities + fa_map
        else:
            raise NotImplementedError(interaction)
    output_activities = squash(output_activities)

    # 3. Scale bottom with output_activities
    scaled_bottom = bottom * output_activities

    # 4. Use attention for a clickme loss
    if normalize_output:
        norm = tf.sqrt(
            tf.reduce_sum(
                tf.pow(output_activities, 2),
                axis=[1, 2],
                keep_dims=True))
        self.attention_losses += [
            output_activities / (norm + 1e-12)]
    else:
        self.attention_losses += [output_activities]
    return scaled_bottom

  def batch_norm_relu(
          self,
          inputs,
          training,
          data_format=None,
          use_relu=True,
          _BATCH_NORM_DECAY=0.997,
          _BATCH_NORM_EPSILON=1e-5):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs

  def fixed_padding(self, inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(
        inputs,
        [
            [0, 0],
            [pad_beg, pad_end],
            [pad_beg, pad_end],
            [0, 0]
        ])
    return padded_inputs

  def conv2d_fixed_padding(
          self,
          inputs,
          filters,
          kernel_size,
          strides,
          data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not
    # on the dimensions of `inputs` (as opposed to using
    # `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = self.fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)

  def building_block(
          self,
          inputs,
          filters,
          training,
          projection_shortcut,
          strides,
          data_format,
          feature_attention=False,
          block_id=None):
    if self.apply_to == 'input':
        if feature_attention == 'paper':
            inputs = self.feature_attention(
                bottom=inputs,
                name=block_id,
                training=training)
        elif feature_attention == 'fc':
            inputs = self.feature_attention_fc(
                bottom=inputs,
                name=block_id,
                training=training,
                squash=self.squash,
                extra_convs=self.extra_convs)
    shortcut = inputs
    inputs = self.batch_norm_relu(inputs, training, data_format)

    # The projection shortcut should come after the first batch norm and
    # ReLU since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = self.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = self.batch_norm_relu(inputs, training, data_format)
    inputs = self.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    # Feature attention applied to the dense path
    if self.apply_to == 'output':
        if feature_attention == 'paper':
            inputs = self.feature_attention(
                bottom=inputs,
                name=block_id,
                training=training)
        elif feature_attention == 'fc':
            inputs = self.feature_attention_fc(
                bottom=inputs,
                name=block_id,
                training=training,
                squash=self.squash,
                extra_convs=self.extra_convs)

    return inputs + shortcut

  def bottleneck_block(
          self,
          inputs,
          filters,
          training,
          projection_shortcut,
          strides,
          data_format,
          feature_attention=False,
          block_id=None):
    if self.apply_to == 'input':
      if feature_attention == 'paper':
        inputs = self.feature_attention(
          bottom=inputs,
          name=block_id,
          training=training)
      elif feature_attention == 'fc':
        inputs = self.feature_attention_fc(
          bottom=inputs,
          name=block_id,
          training=training,
          squash=self.squash,
          extra_convs=self.extra_convs)

    shortcut = inputs
    inputs = self.batch_norm_relu(inputs, training, data_format)

    # The projection shortcut should come after the first batch norm and
    # ReLU since it performs a 1x1 convolution.
    if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)

    inputs = self.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = self.batch_norm_relu(inputs, training, data_format)
    inputs = self.conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = self.batch_norm_relu(inputs, training, data_format)
    inputs = self.conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    # Feature attention applied to the dense path
    if self.apply_to == 'output':
      if feature_attention == 'paper':
          inputs = self.feature_attention(
                bottom=inputs,
                name=block_id,
                training=training)
      elif feature_attention == 'fc':
          inputs = self.feature_attention_fc(
                bottom=inputs,
                name=block_id,
                training=training,
                squash=self.squash,
                extra_convs=self.extra_convs)
    return inputs + shortcut

  def block_layer(
          self,
          inputs,
          filters,
          block_fn,
          blocks,
          strides,
          training,
          name,
          data_format,
          feature_attention):
    filters_out = 4 * filters \
        if self.resnet_size >= 50 else filters

    def projection_shortcut(inputs):
      return self.conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)

    # Only first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(
      inputs=inputs,
      filters=filters,
      training=training,
      projection_shortcut=projection_shortcut,
      strides=strides,
      data_format=data_format,
      feature_attention=feature_attention[0],
      block_id='%s_0' % name)

    for idx in range(1, blocks):
      inputs = block_fn(
        inputs=inputs,
        filters=filters,
        training=training,
        projection_shortcut=None,
        strides=1,
        data_format=data_format,
        feature_attention=feature_attention[idx],
        block_id='%s_%s' % (name, idx))
    return tf.identity(inputs, name)

    # def conv_layer(
    #         self,
    #         bottom,
    #         in_channels=None,
    #         out_channels=None,
    #         name=None,
    #         training=True,
    #         stride=1,
    #         filter_size=3):
    #   """Method for creating a convolutional layer."""
    #   assert name is not None, 'Supply a name for your operation.'
    #   if in_channels is None:
    #     in_channels = int(bottom.get_shape()[-1])
    #   with tf.variable_scope(name):
    #     filt, conv_biases = self.get_conv_var(
    #           filter_size=filter_size,
    #           in_channels=in_channels,
    #           out_channels=out_channels,
    #           name=name)
    #     conv = tf.nn.conv2d(
    #           bottom,
    #           filt,
    #           [1, stride, stride, 1],
    #           padding='SAME')
    #     bias = tf.nn.bias_add(conv, conv_biases)
    #     return bias

    # def get_conv_var(
    #     self,
    #     filter_size,
    #     in_channels,
    #     out_channels,
    #     name,
    #     init_type='xavier'):
    #   if init_type == 'xavier':
    #     weight_init = [
    #       [filter_size, filter_size, in_channels, out_channels],
    #       tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
    #   else:
    #     weight_init = tf.truncated_normal(
    #       [filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
    #   bias_init = tf.truncated_normal([out_channels], .0, .001)
    #   filters = self.get_var(weight_init, name, 0, name + "_filters")
    #   biases = self.get_var(bias_init, name, 1, name + "_biases")

    #   return filters, biases

  def fc_layer(
      self,
      bottom,
      in_size=None,
      out_size=None,
      name=None,
      activation=True,
      training=True):

    # bottom = tf.Print(bottom, [bottom], "fc layer {}".format(name), summarize=20)

    return tf.contrib.layers.fully_connected(bottom, out_size, scope=name)

    # """Method for creating a fully connected layer."""
    # assert name is not None, 'Supply a name for your operation.'
    # if in_size is None:
    #   in_size = int(bottom.get_shape()[-1])
    # with tf.variable_scope(name):
    #   weights, biases = self.get_fc_var(in_size, out_size, name)
    #   x = tf.reshape(bottom, [-1, in_size])
    #   fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
    #   return fc

  # def get_fc_var(
  #         self,
  #         in_size,
  #         out_size,
  #         name):
  #   # if init_type == 'xavier':
  #   #   weight_init = [[in_size, out_size], tf.contrib.layers.xavier_initializer(uniform=False)]
  #   # else:
  #   #   weight_init = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
  #   # bias_init = tf.truncated_normal([out_size], .0, .001)
  #   # weights = self.get_var(weight_init, name, 0, name + "_weights")
  #   # biases = self.get_var(bias_init, name, 1, name + "_biases")

  #   weights = tf.get_variable(
  #     name=name + "_weights",
  #     shape=[in_size, out_size],
  #     # initializer=tf.contrib.layers.xavier_initializer(uniform=False),
  #     trainable=self.trainable,
  #     dtype=tf.bfloat16 if self.use_tpu else tf.float32)

  #   biases = tf.get_variable(
  #     name=name + "_biases",
  #     shape=out_size,
  #     # initializer=tf.truncated_normal_initializer(.0, .001),
  #     trainable=self.trainable,
  #     dtype=tf.bfloat16 if self.use_tpu else tf.float32)
  #   return weights, biases

  # def get_var(
  #         self,
  #         initial_value,
  #         name,
  #         idx,
  #         var_name,
  #         in_size=None,
  #         out_size=None):
  #   with tf.control_dependencies(None):
  #     value = initial_value

  #     if type(value) is list:
  #       var = tf.get_variable(
  #             name=var_name,
  #             shape=value[0],
  #             initializer=value[1],
  #             trainable=self.trainable)
  #     else:
  #       var = tf.get_variable(
  #             name=var_name,
  #             initializer=value,
  #             trainable=self.trainable)
  #     # self.var_dict[(name, idx)] = var
  #   return var


def _get_block_sizes(resnet_size):
  """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = (
          'Could not find layers for selected Resnet size.\n'
          'Size received: {}; sizes allowed: {}.'.format(
              resnet_size, choices.keys()))
  raise ValueError(err)


def _get_attention_sizes(resnet_size):
  """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
  choices = {
      18: [False, [False, True], [True, False], False],
      34: [False, [False, True, True, False], False, False],
      50: [False, [False, False, False, False], True, False],
      101: [False, False, True, True],
      152: [False, False, True, True],
      200: [False, False, True, True]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = (
          'Could not find layers for selected Resnet size.\n'
          'Size received: {}; sizes allowed: {}.'.format(
              resnet_size, choices.keys()))
  raise ValueError(err)


def interpret_nl(nl_string):
  """Return the tensorflow nonlinearity referenced in nl_string."""
  if nl_string == 'relu':
    return tf.nn.relu
  elif nl_string == 'sigmoid':
    return tf.sigmoid
  elif nl_string == 'tanh':
    return tf.tanh
  else:
    raise NotImplementedError(nl_string)
