import numpy as np
from numpy import append
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Conv2D, Add, BatchNormalization, Activation, Lambda, Multiply, Conv2DTranspose, \
    Concatenate
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback
import os, time
from imgaug import augmenters as iaa
from scipy.special import softmax

from src.util import load_data
from src.image_augmentation import ImageCropping
from src.loss import detection_loss, classification_loss, joint_loss
from src.config import Config
from src.predict import eval_pre
from src.model import tune_loss_weight, data_prepare, SFCNnetwork, save_model_weights, det_model_compile, \
    cls_model_compile, joint_model_compile, generator_origin, generator_with_aug, generator_without_aug, \
    callback_preparation

weight_decay = 0.005
epsilon = 1e-7

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'Cls_and_Det')
    TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')

    # TODO@alfred(20220116): 1. detecton performance is not good. 2. the loss for classification is wrong.
    weights = tune_loss_weight()

    CROP_SIZE = 64
    BATCH_SIZE = 1
    EPOCHS = Config.epoch
    TRAIN_STEP_PER_EPOCH = 20
    NUM_TO_CROP, NUM_TO_AUG = 20, 10

    data = data_prepare(DATA_DIR, print_input_shape=True, print_image_shape=True)
    network = SFCNnetwork(l2_regularizer=weights[-1])
    optimizer = SGD(lr=Config.lr, momentum=0.9, decay=1e-6, nesterov=True)
    # optimizer = Adam(lr=1e-3, decay=1e-4)
    model_weights_saver = save_model_weights('base', str(EPOCHS), dir_path=os.path.join(ROOT_DIR, 'model_weights'))

    # Train Detection Branch
    if not os.path.exists(model_weights_saver[0]):
        # if True:
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
                                validation_data=generator_origin(data[3], data[4], data[5],
                                                                 batch_size=BATCH_SIZE,
                                                                 type='detection'),
                                validation_steps=int(np.ceil(data[3].shape[0] / BATCH_SIZE)),
                                callbacks=callback_preparation(det_model, TENSORBOARD_DIR, CHECKPOINT_DIR),
                                workers=4)

        det_model.save_weights(model_weights_saver[0])

    test_img, test_det, test_cls = data[6], data[7], data[8]

    # for im, la in generator_origin(test_img, test_det, test_cls,
    #                                batch_size=test_img.shape[0], type='detection'):
    #     det_pre = det_model.predict(im)
    #     det_pre = softmax(det_pre, axis=-1)
    #     p, r = eval_pre(la, det_pre, prob_thresh=0.5)
    #     print('detection: precision: %.3f, recall: %.3f' % (p, r))
    #     break

    # Train Classification Branch
    # if not os.path.exists(model_weights_saver[1]):
    if True:
        print('classification model is training')
        cls_model = cls_model_compile(nn=network, cls_loss_weight=weights[1],
                                      optimizer=optimizer,
                                      load_weights=model_weights_saver[0],
                                      summary=True)
        cls_model.fit_generator(generator_with_aug(data[0], data[1], data[2],
                                                   crop_size=CROP_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
                                                   type='detection'),
                                epochs=EPOCHS,
                                steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                                validation_data=generator_origin(data[3], data[4], data[5],
                                                                 batch_size=BATCH_SIZE,
                                                                 type='detection'),
                                validation_steps=int(np.ceil(data[3].shape[0] / BATCH_SIZE)),
                                callbacks=callback_preparation(cls_model, TENSORBOARD_DIR, CHECKPOINT_DIR),
                                workers=4)
        cls_model.save_weights(model_weights_saver[1])

    for im, la in generator_origin(test_img, test_det, test_cls,
                                   batch_size=test_img.shape[0], type='detection'):
        cls_pre = cls_model.predict(im)
        cls_pre = softmax(cls_pre, axis=-1)
        p, r = eval_pre(la, cls_pre, prob_thresh=0.5)
        print('classification: precision: %.3f, recall: %.3f' % (p, r))
        break

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
                                  validation_data=generator_origin(data[3], data[4], data[5],
                                                                   batch_size=BATCH_SIZE,
                                                                   type='joint'),
                                  validation_steps=int(np.ceil(data[3].shape[0] / BATCH_SIZE)),
                                  callbacks=callback_preparation(joint_model, TENSORBOARD_DIR, CHECKPOINT_DIR),
                                  workers=4)
        joint_model.save_weights(model_weights_saver[2])

    # test cls model
