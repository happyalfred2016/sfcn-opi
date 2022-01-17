import numpy as np
from keras.optimizers import SGD
import os
import matplotlib.pyplot as plt
from scipy.special import softmax
from config import Config
from model import data_prepare, SFCNnetwork, save_model_weights, det_model_compile, \
    generator_without_aug, generator_with_aug, generator_origin, tune_loss_weight
from metric import non_max_suppression

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

    val_loss, val_acc = det_model.evaluate_generator(
        generator_origin(data[6], data[7], data[8], batch_size=BATCH_SIZE, type='detection'),
        steps=int(np.ceil(data[6].shape[0] / BATCH_SIZE)))
    # det_model.metrics_names

    count = 0
    for im, la in generator_origin(data[6], data[7], data[8],
                                   batch_size=BATCH_SIZE, type='detection'):
        im_in = im[0:1]
        la_in = la[0].squeeze()
        pre = det_model.predict(im_in)
        pre = softmax(pre, axis=-1)[0, :, :, 1]
        b = non_max_suppression(pre, overlap_thresh=0.3, max_boxes=1200, r=15, prob_thresh=0.65)

        plt.figure(figsize=(12, 3))
        plt.subplot(141)
        plt.imshow(im_in[0].astype(np.uint8))

        plt.subplot(142)
        plt.imshow((pre * 255).astype(np.uint8))

        plt.subplot(143)
        # plt.imshow(im_in[0].astype(np.uint8), alpha=0.7)
        # plt.imshow((la_in * 255).astype(np.uint8), alpha=la_in)
        plt.imshow((pre * 255).astype(np.uint8))
        plt.scatter((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, s=15, alpha=0.4, c='tab:red')

        plt.subplot(144)
        plt.imshow((la_in * 255).astype(np.uint8), alpha=la_in)
        plt.scatter((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, s=15, alpha=0.4, c='tab:red')

        plt.show()

        count += 1
        if count > 4:
            break
