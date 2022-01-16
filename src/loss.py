import keras.backend as K
import tensorflow as tf
import numpy as np

epsilon = 1e-7
cls_threshold = 0.8


def detection_loss(weight):
    """
    Detection loss for detection branch.
    :param weight: detection weight.
    """

    def _detection_loss(y_true, y_pred):
        _y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
        weights = tf.stop_gradient((1 - y_true) * weight[0] + y_true * weight[1])
        result = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred, weights=weights)
        return result

    return _detection_loss


def classification_loss(weights, threshold=cls_threshold):
    """
    Classification loss for classification branch.
    :param weights: classification weight for each type of cells.
    :param threshold: default threshold is 0.8 according to paper.
    """

    def _classification_loss(y_true, y_pred):
        indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
        indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
        class_weights = tf.convert_to_tensor(weights, name='cls_weight_convert')
        class_weights = tf.cast(class_weights, tf.float32)
        logits = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        logits = tf.cast(logits, tf.float32, name='logits_cast')
        loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
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
            weight = tf.convert_to_tensor([det_weights[0], det_weights[0], det_weights[0],
                                           det_weights[0], det_weights[0]])
            weight = tf.cast(weight, tf.int32)

            _y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
            weights = tf.stop_gradient(
                tf.reshape(tf.map_fn(lambda x: weight[x], tf.reshape(_y_true, [-1])), tf.shape(y_true)))
            weights = tf.cast(weights, tf.float32)
            result = tf.losses.sparse_softmax_cross_entropy(_y_true, y_pred, weights=weights)
            return result

        def _classification_loss(y_true, y_pred, cls_joint_weights, threshold):
            indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
            indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
            class_weights = tf.convert_to_tensor(cls_joint_weights, name='cls_weight_convert')
            class_weights = tf.cast(class_weights, tf.float32)
            logits = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            logits = tf.cast(logits, tf.float32, name='logits_cast')
            loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
            return loss

        det_loss = _detection_loss(y_true, y_pred, det_weights)
        # det_loss=0.
        cls_loss = _classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss

    return _joint_loss
