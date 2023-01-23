import tensorflow as tf 
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.python.framework import constant_op


##############################################################################################################
########################################### Classification ###################################################
##############################################################################################################

class MaxCriticalSuccessIndex(tf.keras.metrics.Metric):
    """ 
    Calcualte the element-wise [e.g., pixel-wise],
    maximum Critical Success Index out of 20 thresholds.
    If update_state is called more than once, it will add give you the new max 
    csi over all calls. In other words, it accumulates the TruePositives, 
    FalsePositives and FalseNegatives. 
    
    If you want to use it on a fresh y_true,y_pred, make sure you run:

    metric.reset_state()

    then 

    metric(y_true,y_pred) #<-- this will also store the counts of TP etc.

    The shapes of y_true and y_pred should match 
    
    """ 

    def __init__(self, name="max_csi",
                 tp = tf.keras.metrics.TruePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 fp = tf.keras.metrics.FalsePositives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 fn = tf.keras.metrics.FalseNegatives(thresholds=np.arange(0.05,1.05,0.05).tolist()),
                 **kwargs):
        super(MaxCriticalSuccessIndex, self).__init__(name=name, **kwargs)

        #initialize csi value, if no data given, it will be 0 
        self.csi = self.add_weight(name="csi", initializer="zeros")

        #store defined metric functions
        self.tp = tp 
        self.fp = fp 
        self.fn = fn
        #flush functions just in case 
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        if (len(y_true.shape[1:]) > 2 ) and (y_true.shape[-1] == 2):
            #convert back to 1 map 
            y_true = tf.where(y_true[:,:,:,1]>0,1,0)
            #ypred[:,:,:,0] = 1 - y_pred[:,:,:,1]
            y_pred = y_pred[:,:,:,1]
            #ravel for pixelwise comparison
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        #if the output is a map (batch,nx,ny,nl) ravel it
        elif (len(y_true.shape[1:]) > 2):
            y_true = tf.experimental.numpy.ravel(y_true)
            y_pred = tf.experimental.numpy.ravel(y_pred)
        

        #call vectorized stats metrics, add them to running amount of each
        self.tp.update_state(y_true,y_pred)
        self.fp.update_state(y_true,y_pred)
        self.fn.update_state(y_true,y_pred)

        #calc current max csi (so we can get updates per batch)
        self.csi_val = tf.reduce_max(self.tp.result()/(self.tp.result() + self.fn.result() + self.fp.result()))

        #assign the value to the csi 'weight'
        self.csi.assign(self.csi_val)
      
    def result(self):
        return self.csi

    def reset_state(self):
        # Reset the counts 
        self.csi.assign(0.0)
        self.tp.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()

##############################################################################################################
##############################################################################################################
##############################################################################################################

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """Adapted tf.keras.backend.binary_crossentropy to have weights that work for a UNET
    
    Last I checked, weighting the binary cross entropy wouldnt work with the UNETs, it
    use to complain about the shape being not right. 
    
    To get around this, the class here has a way to weight the 1s and 0s differently.
    This is important for alot of meteorological instances where the event we are looking 
    to predict is often a 'rare' event. We mean where the pixels == 1 are like 1% of the 
    total number of pixels. 
    
    the expected shape of y_true is [n_sample,nx,ny,1]
    
    """
    def __init__(self,weights=[1.0,1.0],from_logits=False):
        super().__init__()
        #store weights
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.from_logits=from_logits

    def __str__(self):
        return ("WeightedBinaryCrossEntropy()")
    
    def binary_crossentropy(self,target, output):
        """Binary crossentropy between an output tensor and a target tensor.
        Args:
          target: A tensor with the same shape as `output`.
          output: A tensor.
          from_logits: Whether `output` is expected to be a logits tensor.
              By default, we consider that `output`
              encodes a probability distribution.
        Returns:
          A tensor.
        """
        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)

        # Use logits whenever they are available. `softmax` and `sigmoid`
        # activations cache logits on the `output` Tensor.
        if hasattr(output, '_keras_logits'):
            output = output._keras_logits  # pylint: disable=protected-access
            if self.from_logits:
              warnings.warn(
                  '"`binary_crossentropy` received `from_logits=True`, but the `output`'
                  ' argument was produced by a sigmoid or softmax activation and thus '
                  'does not represent logits. Was this intended?"',
                  stacklevel=2)
            self.from_logits = True

        if self.from_logits:
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

        if (not isinstance(output, (tf.__internal__.EagerTensor, tf.Variable)) and
          output.op.type == 'Sigmoid') and not hasattr(output, '_keras_history'):
            # When sigmoid activation function is used for output operation, we
            # use logits from the sigmoid function directly to compute loss in order
            # to prevent collapsing zero when training.
            assert len(output.op.inputs) == 1
            output = output.op.inputs[0]
            return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

        epsilon_ = _constant_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)

        # Compute cross entropy from probabilities.
        bce = target * tf.math.log(output + tf.keras.backend.epsilon())
        bce += (1 - target) * tf.math.log(1 - output + tf.keras.backend.epsilon()) 
        return -bce

    def call(self, y_true, y_pred):
        
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        
        loss = self.binary_crossentropy(y_true, y_pred)
    
        #build weight_matrix 
        ones_array = tf.ones_like(y_true)
        weights_for_nonzero = tf.math.multiply(ones_array,self.w1)
        weights_for_zero = tf.math.multiply(ones_array,self.w2)
        weight_matrix = tf.where(tf.greater(y_true,0.0),weights_for_nonzero,
                           weights_for_zero)
        self.w = weight_matrix
        #weight non-zero pixels more
        loss = tf.math.multiply(weight_matrix,loss)
    
        return tf.reduce_mean(loss)

def _constant_to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    This is slightly faster than the _to_tensor function, at the cost of
    handling fewer cases.
    Args:
        x: An object to be converted (numpy arrays, floats, ints and lists of
            them).
        dtype: The destination type.
    Returns:
        A tensor.
    """
    return constant_op.constant(x, dtype=dtype)