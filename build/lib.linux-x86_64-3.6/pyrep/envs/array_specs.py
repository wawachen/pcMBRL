import numpy as np

class ArraySpec(object):
  """Describes a numpy array or scalar shape and dtype.
  An `ArraySpec` allows an API to describe the arrays that it accepts or
  returns, before that array exists.
  The equivalent version describing a `tf.Tensor` is `TensorSpec`.
  """

  def __init__(self, shape, dtype, name=None):
    """Initializes a new `ArraySpec`.
    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.
    Raises:
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    self._shape = tuple(shape)
    self._dtype = np.dtype(dtype)
    self._name = name

  @property
  def shape(self):
    """Returns a `tuple` specifying the array shape."""
    return self._shape

  @property
  def dtype(self):
    """Returns a numpy dtype specifying the array dtype."""
    return self._dtype

  @property
  def name(self):
    """Returns the name of the ArraySpec."""
    return self._name

  def inform_specs(self):
    return 'ArraySpec(shape={}, dtype={}, name={})'.format(
        self.shape, repr(self.dtype), repr(self.name))

  

# a = ArraySpec(shape = (3,4),dtype = np.float32)
# print(a.inform_specs())





