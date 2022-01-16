import numpy as np
from keras.optimizers import SGD
import os
import matplotlib.pyplot as plt
from scipy.special import softmax
from config import Config
from model import data_prepare, SFCNnetwork, save_model_weights, det_model_compile, \
    generator_without_aug, generator_with_aug, tune_loss_weight

weights = tune_loss_weight()
CROP_SIZE = 64
BATCH_SIZE = 1
EPOCHS = Config.epoch
TRAIN_STEP_PER_EPOCH = 20
NUM_TO_CROP, NUM_TO_AUG = 20, 10

data = data_prepare(print_input_shape=True, print_image_shape=True)
network = SFCNnetwork(l2_regularizer=weights[-1])
optimizer = SGD(lr=Config.lr, momentum=0.9, decay=1e-6, nesterov=True)

model_weights_saver = save_model_weights('base', str(EPOCHS))
# Train Detection Branch

test_img, test_det, test_cls = data[6], data[7], data[8]

if os.path.exists(model_weights_saver[0]):
    det_model = det_model_compile(nn=network, det_loss_weight=weights[0], optimizer=optimizer, softmax_trainable=False)
    det_model.load_weights(model_weights_saver[0])

    # det_model.evaluate_generator(generator_with_aug(data[3], data[4], data[5],
    #                                                 batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
    #                                                 crop_num=NUM_TO_CROP, aug_num=NUM_TO_AUG,
    #                                                 type='detection'),
    #                              steps=5)
    # det_model.metrics_names

    for im, la in generator_with_aug(data[3], data[4], data[5],
                                     batch_size=BATCH_SIZE, crop_size=CROP_SIZE,
                                     crop_num=NUM_TO_CROP, aug_num=10,
                                     type='detection'):
        im_in = im[0:1]
        la_in = la[0]
        plt.subplot(121)
        plt.imshow(im_in[0].astype(np.uint8))
        plt.subplot(122)
        plt.imshow((la_in.squeeze() * 255).astype(np.uint8))
        plt.show()



        pre = det_model.predict(im_in)
        pre = softmax(pre, axis=-1)[0, :, :, 1]

        plt.subplot(121)
        plt.imshow(im_in[0].astype(np.uint8))
        plt.subplot(122)
        plt.imshow((pre * 255).astype(np.uint8))
        plt.show()

        break

