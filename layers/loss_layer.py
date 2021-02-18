import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(prediction, target, weights=None, alpha=0.25, gamma=2):
    """Compute focal loss for predictions.
    Multi-labels Focal loss formula:
        FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
             ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
    prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
    target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
    weights: A float tensor of shape [batch_size, num_anchors]
    alpha: A scalar tensor for focal loss alpha hyper-parameter
    gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
    loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = prediction
    print("Focal loss alpha=%f, gamma=%f" % (alpha, gamma))
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)  # Get a zero tensor with the given dtype.
    ones = tf.ones_like(sigmoid_p, dtype=sigmoid_p.dtype)  # tf.constant(1.0, shape = prediction.get_shape())
    pos_p_sub = array_ops.where(target > zeros, ones - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target > zeros, zeros, sigmoid_p)
    
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(
        tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return per_entry_cross_ent
