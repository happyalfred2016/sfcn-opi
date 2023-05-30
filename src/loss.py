import keras.backend as K
import tensorflow as tf

epsilon = 1e-7
cls_threshold = 0.8


def _gen_weight(y_true, weight):
    weights = tf.convert_to_tensor(weight, dtype=tf.float32)
    #
    tensor_w = tf.zeros_like(y_true, dtype='float32')
    for i in range(weights.shape[0]):
        tensor_w += tf.cast(tf.equal(y_true, i), tf.float32) * weights[i]
    return tensor_w
    # c0 = tf.cast(tf.equal(y_true, 0), tf.float32) * weights[0]
    # c1 = tf.cast(tf.equal(y_true, 1), tf.float32) * weights[1]
    # c2 = tf.cast(tf.equal(y_true, 2), tf.float32) * weights[2]
    # c3 = tf.cast(tf.equal(y_true, 3), tf.float32) * weights[3]
    # c4 = tf.cast(tf.equal(y_true, 4), tf.float32) * weights[4]
    # return c0 + c1 + c2 + c3 + c4


def detection_loss(weight):
    """
    Detection loss for detection branch.
    :param weight: detection weight.
    """

    def _detection_loss(y_true, y_pred):
        _y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
        # y_true = tf.cast(y_true, tf.int32)
        # weights = tf.stop_gradient((1 - y_true) * weight[0] + y_true * weight[1])
        weights = tf.stop_gradient(_gen_weight(_y_true, weight))
        result = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred, weights=weights)
        return result

    return _detection_loss


def classification_loss(weight, threshold=cls_threshold):
    """
    Classification loss for classification branch.
    :param weights: classification weight for each type of cells.
    :param threshold: default threshold is 0.8 according to paper.
    """

    def _classification_loss(y_true, y_pred):
        _y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
        # y_true = tf.cast(y_true, tf.int32)
        # indicator = tf.greater_equal(K.sum(K.softmax(y_pred, axis=-1)[:, :, :, 1:], axis=-1, keepdims=True),
        #                              threshold, name='indicator_great')
        # indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
        # weights = tf.stop_gradient(_gen_weight(y_true, weight)) * indicator
        weights = tf.stop_gradient(_gen_weight(_y_true, weight))
        loss = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred, weights=weights)
        # loss = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred)
        return loss

    return _classification_loss


def joint_loss(det_weights, cls_joint_weights, joint_weights, cls_threshold=cls_threshold):
    """
    Joint loss for joint model.
    :param det_weights: detection weight, is same as we use in detection branch.
    :param cls_joint_weights: classification weights, different from classification weight in cls branch.
    :param joint_weights: joint weights, adds on classification loss part.
    :param cls_threshold: cls threshold is default setting to 0.8
    """

    def _joint_loss(y_true, y_pred):
        def _detection_loss(y_true, y_pred, det_weights):
            weights = tf.stop_gradient(_gen_weight(y_true, [det_weights[0],
                                                            det_weights[1],
                                                            det_weights[1],
                                                            det_weights[1],
                                                            det_weights[1]]))

            _y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
            result = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred, weights=weights)
            return result

        def _classification_loss(y_true, y_pred, cls_joint_weights, threshold):
            _y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
            indicator = tf.greater_equal(K.sum(K.softmax(y_pred, axis=-1)[:, :, :, 1:], axis=-1, keepdims=True),
                                         threshold, name='indicator_great')
            indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
            # weights = tf.stop_gradient(_gen_weight(y_true, cls_joint_weights)) * indicator
            weights = tf.stop_gradient(_gen_weight(y_true, cls_joint_weights))
            loss = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred, weights=weights)

            return loss

        det_loss = _detection_loss(y_true, y_pred, det_weights)
        # det_loss=0.
        cls_loss = _classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss

    return _joint_loss
