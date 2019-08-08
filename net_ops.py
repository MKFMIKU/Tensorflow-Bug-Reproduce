import tensorflow as tf
from tensorflow.python.framework import ops

cutoms_ops = tf.load_op_library('./lib/ops.so')

# -- Register operations ------------------------------------------------------
bilateral_slice = cutoms_ops.bilateral_slice
bilateral_slice_apply = cutoms_ops.bilateral_slice_apply

# ----------- Register gradients ----------------------------------------------
@ops.RegisterGradient('BilateralSlice')
def _bilateral_slice_grad(op, grad):
  grid_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  return cutoms_ops.bilateral_slice_grad(grid_tensor, guide_tensor, grad)


@ops.RegisterGradient('BilateralSliceApply')
def _bilateral_slice_grad(op, grad):
  grid_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  input_tensor = op.inputs[2]
  has_offset = op.get_attr('has_offset')
  return cutoms_ops.bilateral_slice_apply_grad(
      grid_tensor, guide_tensor, input_tensor, grad, has_offset=has_offset)

# ----------- Register Shape inference ----------------------------------------
@ops.RegisterShape('BilateralSlice')
def _bilateral_slice_shape(op):
  input_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  return [guide_tensor.get_shape().concatenate(input_tensor.get_shape()[-1])]


@ops.RegisterShape('BilateralSliceApply')
def _bilateral_slice_shape(op):
  grid_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  input_tensor = op.inputs[2]

  has_offset = op.get_attr('has_offset')
  chan_in = input_tensor.get_shape()[-1]
  chan_grid = grid_tensor.get_shape()[-1]

  if has_offset:
    chan_out = chan_grid // (chan_in+1)
  else:
    chan_out = chan_grid // chan_in
  return [guide_tensor.get_shape().concatenate(chan_out)]
