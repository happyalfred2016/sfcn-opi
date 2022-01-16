import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, Lambda, Multiply, Conv2DTranspose, \
    Concatenate
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback
from util import load_data
import os, time
from image_augmentation import ImageCropping
from imgaug import augmenters as iaa
from loss import detection_loss, classification_loss, joint_loss
from config import Config

weight_decay = 0.005
epsilon = 1e-7

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'Cls_and_Det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')


class Conv3l2(keras.layers.Conv2D):
    """
    Custom convolution layer with default 3*3 kernel size and L2 regularization.
    Default padding change to 'same' in this case.
    """

    def __init__(self, filters, kernel_regularizer_weight,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(self.__class__, self).__init__(filters,
                                             kernel_size=(3, 3),
                                             strides=strides,
                                             padding=padding,
                                             data_format=data_format,
                                             dilation_rate=dilation_rate,
                                             activation=activation,
                                             use_bias=use_bias,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             kernel_regularizer=keras.regularizers.l2(kernel_regularizer_weight),
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             kernel_constraint=kernel_constraint,
                                             bias_constraint=bias_constraint,
                                             **kwargs)


class SFCNnetwork:
    """
    Backbone of SFCN-OPI.
    """

    def __init__(self, l2_regularizer, input_shape=(None, None, 3)):
        # self.inputs = inputs
        self.input_shape = input_shape
        self.l2r = l2_regularizer

    def input_normalize(self, inputs):
        # img = image / 255.
        # img -= np.mean(img, keepdims=True)
        # img /= (np.std(img, keepdims=True) + 1e-7)

        x = keras.layers.Lambda(lambda x: x / 255.)(inputs)
        x = keras.layers.Lambda(lambda x: x - K.mean(x, keepdims=True))(x)
        x = keras.layers.Lambda(lambda x: x / (K.std(x, keepdims=True) + 1e-7))(x)
        return x

    def first_layer(self, inputs, trainable=True):
        """
        First convolution layer.
        """
        x = Conv3l2(filters=32, name='Conv_1',
                    kernel_regularizer_weight=self.l2r,
                    trainable=trainable)(inputs)
        x = BatchNormalization(name='BN_1', trainable=trainable)(x)
        x = Activation('relu', name='act_1', trainable=trainable)(x)
        return x

    ###########################################
    # ResNet Graph
    ###########################################
    def identity_block(self, f, stage, block, inputs, trainable=True):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        :param trainable: freeze layer if false
        """
        x_shortcut = inputs

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(block) + '_' + str(stage) + '_idblock_conv_1',
                    trainable=trainable)(inputs)
        x = BatchNormalization(name=str(block) + '_' + str(stage) + '_idblock_BN_1',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(block) + '_' + str(stage) + '_idblock_act_1',
                       trainable=trainable)(x)

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(block) + '_' + str(stage) + '_idblock_conv_2')(x)
        x = BatchNormalization(name=str(block) + '_' + str(stage) + '_idblock_BN_2',
                               trainable=trainable)(x)
        x = Add(name=str(block) + '_' + str(stage) + '_idblock_add', trainable=trainable)([x, x_shortcut])
        x = Activation('relu', name=str(block) + '_' + str(stage) + '_idblock_act_2',
                       trainable=trainable)(x)
        return x

    def convolution_block(self, f, stage, block, inputs, trainable=True):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x = Conv3l2(filters=f, strides=(2, 2), kernel_regularizer_weight=self.l2r,
                    name=str(block) + '_' + str(stage) + '_convblock_conv_1',
                    trainable=trainable)(inputs)
        x = BatchNormalization(name=str(block) + '_' + str(stage) + '_convblock_BN_1',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(block) + '_' + str(stage) + '_convblock_act_1',
                       trainable=trainable)(x)
        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(block) + '_' + str(stage) + '_convblock_conv_2',
                    trainable=trainable)(x)
        x = BatchNormalization(name=str(block) + '_' + str(stage) + '_convblock_BN_2',
                               trainable=trainable)(x)

        x_shortcut = Conv2D(f, kernel_size=(1, 1), strides=(2, 2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name=str(block) + '_' + str(stage) + '_convblock_shortcut_conv',
                            trainable=trainable)(inputs)
        x_shortcut = BatchNormalization(name=str(block) + '_' + str(stage) + '_convblock_shortcut_BN_1',
                                        trainable=trainable)(x_shortcut)
        x = Add(name=str(block) + '_' + str(stage) + '_convblock_add',
                trainable=trainable)([x, x_shortcut])
        x = Activation('relu', name=str(block) + '_' + str(stage) + '_convblock_merge_act',
                       trainable=trainable)(x)
        return x

    def res_block(self, inputs, filter, stages, block, trainable=True, if_conv=False):
        x = inputs
        if not if_conv:  # if_conv is False
            for stage in range(stages):
                x = self.identity_block(f=filter, stage=stage, block=block,
                                        inputs=x, trainable=trainable)
        else:
            for stage in range(stages):
                if stage == 0:
                    x = self.convolution_block(f=filter, stage=stage, block=block, inputs=inputs,
                                               trainable=trainable)
                else:
                    x = self.identity_block(f=filter, stage=stage, block=block, inputs=x,
                                            trainable=trainable)
        return x

    ######################
    # FCN BACKBONE
    # ####################
    def first_and_second_res_blocks(self, inputs, first_filter, second_filter, trainable=True):
        """
        Shared residual blocks for detection and classification layers.
        """
        x = self.res_block(inputs, filter=first_filter, stages=9, block=1,
                           trainable=trainable)
        x = self.res_block(x, filter=second_filter, stages=9, block=2, if_conv=True,
                           trainable=trainable)
        return x

    def share_layer(self, input, trainable=True):
        with tf.variable_scope("shared_layer"):
            x = self.first_layer(input, trainable=trainable)
            x_future_det_one = self.first_and_second_res_blocks(x, 32, 64, trainable=trainable)
            x_future_cls_det_two = self.res_block(x_future_det_one, filter=128, stages=9, block=3, if_conv=True,
                                                  trainable=trainable)
            # print(tf.trainable_variables())
        return x_future_det_one, x_future_cls_det_two

    ###################
    # Detection Branch
    ###################
    def detection_branch_wrapper(self, input_one, input_two, trainable=True, softmax_trainable=False):
        x_divergent_one = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                                 name='conv2D_diverge_one',
                                 trainable=trainable)(input_one)
        x_divergent_one = BatchNormalization(name='bn_diverge_one',
                                             trainable=trainable)(x_divergent_one)
        x_divergent_one = Activation('relu', trainable=trainable)(x_divergent_one)

        x_divergent_two = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(self.l2r),
                                 name='conv_diverge_two',
                                 trainable=trainable)(input_two)
        x_divergent_two = BatchNormalization(name='bn_diverge_two',
                                             trainable=trainable)(x_divergent_two)
        x_divergent_two = Activation('relu',
                                     trainable=trainable)(x_divergent_two)

        x_divergent_two = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                          kernel_regularizer=keras.regularizers.l2(self.l2r),
                                          name='deconv_before_summation',
                                          trainable=trainable)(x_divergent_two)
        x_divergent_two = BatchNormalization(name='bn_deconv_diverge_two',
                                             trainable=trainable)(x_divergent_two)
        x_divergent_two = Activation('relu', name='last_detection_act',
                                     trainable=trainable)(x_divergent_two)

        x_merge = Add(name='merge_two_divergence',
                      trainable=trainable)([x_divergent_one, x_divergent_two])
        x_detection = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                      kernel_regularizer=keras.regularizers.l2(self.l2r),
                                      name='Deconv_detection_final_layer',
                                      trainable=trainable)(x_merge)
        x_detection = BatchNormalization(name='last_detection_bn',
                                         trainable=trainable)(x_detection)
        # The detection output
        if softmax_trainable == True:
            x_detection = Activation('softmax', name='Detection_output',
                                     trainable=trainable)(x_detection)
        return x_detection

    def detection_branch(self, trainable=True, softmax_trainable=False):
        input_img = Input(shape=self.input_shape)

        nimg = self.input_normalize(input_img)
        x_future_det_one, x_future_cls_det_two = self.share_layer(nimg, trainable=trainable)
        # The detection output
        x_detection = self.detection_branch_wrapper(x_future_det_one, x_future_cls_det_two,
                                                    softmax_trainable=softmax_trainable)
        # The classification output
        det_model = Model(inputs=input_img,
                          outputs=x_detection)
        return det_model

    ############################
    # Classification Branch
    ############################
    def classification_branch_wrapper(self, input, softmax_trainable=False):
        x = self.res_block(input, filter=128, stages=9, block=4)
        # all layers before OPI
        x = Conv2D(filters=5, kernel_size=(1, 1), padding='same', name='conv2d_after_fourth_resblock',
                   kernel_regularizer=keras.regularizers.l2(self.l2r))(x)
        x = BatchNormalization(name='bn_after_fourth_resblock')(x)
        x = Activation('relu', name='relu_after_fourth_resblock')(x)
        x = Conv2DTranspose(filters=5, kernel_size=(3, 3),
                            strides=(2, 2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name='secondlast_deconv_before_cls')(x)
        x = BatchNormalization(name='secondlast_bn_before_cls')(x)
        x = Activation('relu', name='last_relu_before_cls')(x)
        x = Conv2DTranspose(filters=5, kernel_size=(3, 3),
                            strides=(2, 2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name='last_deconv_before_cls')(x)
        x_output = BatchNormalization(name='last_bn_before_cls')(x)
        if softmax_trainable == True:
            x_output = Activation('softmax', name='Classification_output')(x_output)
        return x_output

    def classification_branch(self, trainable=False, softmax_trainable=False):
        """classification branch, seprate from detection branch.
        """
        input_img = Input(shape=self.input_shape)
        nimg = self.input_normalize(input_img)
        # this shared layer is frozen before joint training
        x_useless, x_cls = self.share_layer(nimg, trainable=trainable)
        cls_output = self.classification_branch_wrapper(x_cls, softmax_trainable=softmax_trainable)
        cls_model = Model(inputs=input_img,
                          outputs=cls_output)
        return cls_model

    #########################
    # Joint training
    #########################
    def joint_branch(self, trainable=True, softmax_trainable=False):
        """
        joint branch of detection and classification
        :param trainable: unfreeze detection branch layer if set to true
        """
        input_img = Input(shape=self.input_shape)

        nimg = self.input_normalize(input_img)
        x_future_det_one, x_future_cls_det_two = self.share_layer(nimg, trainable=trainable)
        x_detection = self.detection_branch_wrapper(x_future_det_one, x_future_cls_det_two, trainable=trainable,
                                                    softmax_trainable=softmax_trainable)
        x_classification = self.classification_branch_wrapper(x_future_cls_det_two,
                                                              softmax_trainable=softmax_trainable)
        # TODO@alfred
        # x_detection = K.repeat_elements(x_detection[:, :, :, 1:], rep=x_classification.shape[-1], axis=-1)
        # joint_x = Multiply(name='joint_multiply_layer')([x_detection[:, :, :, 1:], x_classification])

        temp_x = keras.layers.Lambda(lambda x: K.repeat_elements(x[:, :, :, 1:], rep=5, axis=-1))(x_detection)
        joint_x = Multiply(name='joint_multiply_layer')([temp_x, x_classification])

        # joint_x = keras.layers.Lambda(
        #     lambda x: x[0][:, :, :, 1:] * x[1], name='joint_multiply_layer')([x_detection, x_classification])

        # TODO@alfred: should not be the original input_img???? WTF?
        # input_img = Input(shape=self.input_shape)

        joint_model = Model(inputs=input_img,
                            outputs=joint_x)
        return joint_model


class TimerCallback(Callback):
    """Tracking time spend on each epoch as well as whole training process.
    """

    def __init__(self):
        super(TimerCallback, self).__init__()
        self.epoch_time = 0
        self.training_time = 0

    def on_train_begin(self, logs=None):
        self.training_time = time.time()

    def on_train_end(self, logs=None):
        end_time = np.round(time.time() - self.training_time, 2)
        time_to_min = end_time / 60
        print('training takes {} minutes'.format(time_to_min))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print('epoch takes {} seconds to train'.format(np.round(time.time() - self.epoch_time), 2))


def data_prepare(print_image_shape=False, print_input_shape=False):
    """
    prepare data for model.
    :param print_image_shape: print image shape if set true.
    :param print_input_shape: print input shape(after categorize) if set true
    :return: list of input to model
    """

    def reshape_mask(origin, cate, num_class):
        return cate.reshape((origin.shape[0], origin.shape[1], origin.shape[2], num_class))

    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train')
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation')
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test')

    if print_image_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_det_masks: {}, train_cls_masks: {}'.format(train_imgs.shape, train_det_masks.shape,
                                                                                train_cls_masks.shape))
        print(
            'valid_imgs: {}, valid_det_masks: {}, validn_cls_masks: {}'.format(valid_imgs.shape, valid_det_masks.shape,
                                                                               valid_cls_masks.shape))
        print('test_imgs: {}, test_det_masks: {}, test_cls_masks: {}'.format(test_imgs.shape, test_det_masks.shape,
                                                                             test_cls_masks.shape))
        print()

    train_det, train_cls = np.expand_dims(train_det_masks, axis=-1), np.expand_dims(train_cls_masks, axis=-1)
    valid_det, valid_cls = np.expand_dims(valid_det_masks, axis=-1), np.expand_dims(valid_cls_masks, axis=-1)
    test_det, test_cls = np.expand_dims(test_det_masks, axis=-1), np.expand_dims(test_cls_masks, axis=-1)

    if print_input_shape:
        print('input shape print below: ')
        print('train_imgs: {}, train_det: {}, train_cls: {}'.format(train_imgs.shape, train_det.shape, train_cls.shape))
        print(
            'valid_imgs: {}, valid_det: {}, validn_cls: {}'.format(valid_imgs.shape, valid_det.shape, valid_cls.shape))
        print('test_imgs: {}, test_det: {}, test_cls: {}'.format(test_imgs.shape, test_det.shape, test_cls.shape))
        print()
    return [train_imgs, train_det, train_cls, valid_imgs, valid_det, valid_cls, test_imgs, test_det, test_cls]


def crop_on_fly(img, det_mask, cls_mask, crop_size):
    """
    Crop image randomly on each training batch
    """
    imgcrop = ImageCropping()
    cropped_img, cropped_det_mask, cropped_cls_mask = imgcrop.crop_image_batch(img, [det_mask, cls_mask],
                                                                               desired_shape=(crop_size, crop_size))
    return cropped_img, cropped_det_mask, cropped_cls_mask


def aug_on_fly(img, det_mask, cls_mask):
    """Do augmentation with different combination on each training batch
    """

    def image_basic_augmentation(image, masks, ratio_operations=0.9):
        # without additional operations
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            sometimes(
                iaa.SomeOf((0, 5), [
                    iaa.Fliplr(hor_flip_angle),
                    iaa.Flipud(ver_flip_angle),
                    iaa.Affine(shear=(-16, 16)),
                    iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                    iaa.PerspectiveTransform(scale=(0.01, 0.1))
                ]))
        ])

        seq_to_deterministic = seq.to_deterministic()
        aug_img, aug_det_mask = seq_to_deterministic(images=image, segmentation_maps=masks[0])
        _aug_img, aug_cls_mask = seq_to_deterministic(images=image, segmentation_maps=masks[1])

        assert np.isclose(_aug_img, aug_img).all()
        return aug_img, aug_det_mask, aug_cls_mask

    aug_image, aug_det_mask, aug_cls_mask = image_basic_augmentation(image=img, masks=[det_mask, cls_mask])
    return aug_image, aug_det_mask, aug_cls_mask


def generator_with_aug(features, det_labels, cls_labels, batch_size, crop_size,
                       type,
                       crop_num=15, aug_num=10):
    """
    generator with basic augmentations which have been in the paper.
    :param features: image.
    :param det_labels: detection mask as label
    :param cls_labels: classification mask as label
    :param batch_size: batch size
    :param crop_size: default size is 64
    :param type: type must be one of detection, classification or joint
    :param crop_num: how many cropped image for a single image.
    :param aug_num: num of augmentation per cropped image
    """
    assert type in ['detection', 'classification', 'joint']
    batch_features = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 3))
    batch_det_labels = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, det_labels.shape[-1]))
    batch_cls_labels = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, cls_labels.shape[-1]))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            for j in range(crop_num):
                feature_index = features[index]
                det_label_index = det_labels[index]
                cls_label_index = cls_labels[index]
                feature, det_label, cls_label = crop_on_fly(feature_index,
                                                            det_label_index,
                                                            cls_label_index,
                                                            crop_size=crop_size)
                for k in range(aug_num):
                    aug_feature, aug_det_label, aug_cls_label = aug_on_fly(feature, det_label, cls_label)
                    batch_features[counter] = aug_feature
                    batch_det_labels[counter] = aug_det_label
                    batch_cls_labels[counter] = aug_cls_label
                    counter = counter + 1
        if type == 'detection':
            yield batch_features, batch_det_labels
        elif type == 'classification' or type == 'joint':
            yield batch_features, batch_cls_labels


def generator_without_aug(features, det_labels, cls_labels, batch_size, crop_size,
                          type,
                          crop_num=25):
    """
    generator without any augmentation, only randomly crop image into [64, 64, channel].
    :param features: image.
    :param det_labels: detection mask as label
    :param cls_labels: classification mask as label
    :param batch_size: batch size
    :param crop_size: default size is 64
    :param crop_num: how many cropped image for a single image.
    """
    batch_features = np.zeros((batch_size * crop_num, crop_size, crop_size, 3))
    batch_det_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, 2))
    batch_cls_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, 5))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            for j in range(crop_num):
                feature, det_label, cls_label = crop_on_fly(features[index],
                                                            det_labels[index], cls_labels[index], crop_size=crop_size)
                batch_features[counter] = feature
                batch_det_labels[counter] = det_label
                batch_cls_labels[counter] = cls_label
                counter += 1
        if type == 'detection':
            yield batch_features, batch_det_labels
        elif type == 'classification' or type == 'joint':
            yield batch_features, batch_cls_labels


def callback_preparation(model):
    """
    implement necessary callbacks into model.
    :return: list of callback.
    """
    timer = TimerCallback()
    timer.set_model(model)
    tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, 'base_tensorboard_logs'))
    checkpoint_callback = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'base_checkpoint',
                                                       'train_point.h5'), save_best_only=True, period=1)
    return [tensorboard_callback, checkpoint_callback, timer]


def det_model_compile(nn, det_loss_weight, softmax_trainable,
                      optimizer, summary=False):
    """

    :param det_loss_weight:
    :param kernel_weight:
    :param summary:
    :return:
    """
    print('detection model is set')
    det_model = nn.detection_branch(softmax_trainable=softmax_trainable)
    det_model.compile(optimizer=optimizer,
                      loss=detection_loss(det_loss_weight),
                      metrics=['accuracy'])
    if summary == True:
        det_model.summary()
    return det_model


def cls_model_compile(nn, cls_loss_weight, load_weights,
                      optimizer, summary=False):
    """

    :param det_loss_weight:
    :param kernel_weight:
    :param summary:
    :return:
    """
    print('classification model is set')
    cls_model = nn.classification_branch(trainable=False)
    cls_model.load_weights(load_weights, by_name=True)
    cls_model.compile(optimizer=optimizer,
                      loss=classification_loss(cls_loss_weight),
                      metrics=['accuracy'])
    if summary == True:
        cls_model.summary()
    return cls_model


def joint_model_compile(nn, det_loss_weight, cls_loss_in_joint, joint_loss_weight,
                        load_weights, optimizer, summary=False):
    """
    :param nn: network
    :param det_loss_weight: detection weight for joint loss.
    :param cls_loss_in_joint: cls weight for joint loss.
    :param joint_loss_weight: regularizer for classification loss part.
    :param load_weights: load weights for this model.
    :param optimizer: optimizer for this model.
    :param summary: if print model summary
    :return:
    """
    print('classification model is set')
    joint_model = nn.joint_branch()
    joint_model.load_weights(load_weights, by_name=True)
    joint_model.compile(optimizer=optimizer,
                        loss=joint_loss(det_loss_weight, cls_loss_in_joint, joint_loss_weight),
                        metrics=['accuracy'])
    if summary == True:
        joint_model.summary()
    return joint_model


def tune_loss_weight():
    """
    use this function to fine tune weights later.
    :return:
    """

    # y_train = [y.argmax() for y in y_train]
    # class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    print('weight initialized')
    cls_weight = np.array([0.5, 0.9, 1.01, 0.68, 1.9])

    # TODO: verify
    det_weight = np.array([1., 100.])
    cls_weight_in_joint = [0.5, 0.83, 0.94, 0.78, 2]
    joint_weight = 1
    kernel_weight = 1
    return [det_weight, cls_weight, cls_weight_in_joint, joint_weight, kernel_weight]


def save_model_weights(type, hyper):
    """
    Set path to save model for each part of the model.
    :param type: type of the model, either detection, classification or joint.
    :param hyper: name with hyper param.
    """
    model_weights = os.path.join(ROOT_DIR, 'model_weights')
    det_model_weights_saver = os.path.join(model_weights,
                                           str(type) + '_model_weights',
                                           hyper + '_' + str(type) + '_det_train_model.h5')
    cls_model_weights_saver = os.path.join(model_weights,
                                           str(type) + '_model_weights',
                                           hyper + '_' + str(type) + '_cls_train_model.h5')
    joint_model_weights_saver = os.path.join(model_weights,
                                             str(type) + '_model_weights',
                                             hyper + '_' + str(type) + '_joint_train_model.h5')
    return [det_model_weights_saver, cls_model_weights_saver, joint_model_weights_saver]


if __name__ == '__main__':
    # TODO@alfred(20220116): 1. detecton performance is not good. 2. the loss for classification is wrong.
    weights = tune_loss_weight()

    CROP_SIZE = 64
    BATCH_SIZE = 1
    EPOCHS = Config.epoch
    TRAIN_STEP_PER_EPOCH = 20
    NUM_TO_CROP, NUM_TO_AUG = 20, 10

    data = data_prepare(print_input_shape=True, print_image_shape=True)
    network = SFCNnetwork(l2_regularizer=weights[-1])
    # optimizer = SGD(lr=Config.lr, momentum=0.9, decay=1e-6, nesterov=True)

    optimizer = Adam(lr=1e-3, decay=1e-4)

    model_weights_saver = save_model_weights('base', str(EPOCHS))

    # Train Detection Branch
    # if not os.path.exists(model_weights_saver[0]):
    det_model = det_model_compile(nn=network, det_loss_weight=weights[0], optimizer=optimizer,
                                  softmax_trainable=False, summary=True)
    print('detection model is training')
    det_model.fit_generator(generator_with_aug(data[0], data[1], data[2],
                                               crop_size=CROP_SIZE,
                                               batch_size=BATCH_SIZE,
                                               crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                               type='detection'),
                            epochs=EPOCHS,
                            steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                            validation_data=generator_with_aug(data[3], data[4], data[5],
                                                               batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
                                                               crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                                               type='detection'),
                            validation_steps=5,
                            callbacks=callback_preparation(det_model), )

    det_model.save_weights(model_weights_saver[0])

    # Train Classification Branch
    if not os.path.exists(model_weights_saver[1]):
        print('classification model is training')
        cls_model = cls_model_compile(nn=network, cls_loss_weight=weights[1],
                                      optimizer=optimizer,
                                      load_weights=model_weights_saver[0])
        cls_model.fit_generator(generator_with_aug(data[0], data[1], data[2],
                                                   crop_size=CROP_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                                   type='detection'),
                                epochs=EPOCHS,
                                steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                validation_data=generator_with_aug(data[3], data[4], data[5],
                                                                   batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
                                                                   crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                                                   type='classification'),
                                validation_steps=5, callbacks=callback_preparation(cls_model))
        cls_model.save_weights(model_weights_saver[1])

    # Train Joint Model
    if not os.path.exists(model_weights_saver[2]):
        print('joint model is training')
        joint_model = joint_model_compile(nn=network, det_loss_weight=weights[0], cls_loss_in_joint=weights[2],
                                          joint_loss_weight=weights[3], optimizer=optimizer,
                                          load_weights=model_weights_saver[1])
        joint_model.fit_generator(generator_with_aug(data[0], data[1], data[2],
                                                     crop_size=CROP_SIZE,
                                                     batch_size=BATCH_SIZE,
                                                     crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                                     type='joint'),
                                  epochs=EPOCHS,
                                  steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                  validation_data=generator_with_aug(data[3], data[4], data[5],
                                                                     batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
                                                                     crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                                                     type='joint'),
                                  validation_steps=5, callbacks=callback_preparation(joint_model))
        joint_model.save_weights(model_weights_saver[2])
