import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import precision_score, recall_score
import cv2
from .metric import non_max_suppression

eps = 1e-6


def truelabel_center_extract(cls_img, cla_list):
    ker = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]])
    fimg = cv2.filter2D(cls_img.astype(float), -1, ker)

    # plt.figure()
    # plt.matshow(cls_img)
    # plt.show()
    #
    # for coors in {i: np.stack(np.where(np.isclose(fimg, i * ker.sum())), axis=1) for i in cla_list}.values():
    #
    #     plt.scatter(coors[:, 1], coors[:,0], s=1)
    #
    # ax = plt.gca()  #
    # ax.xaxis.set_ticks_position('top')  #
    # ax.invert_yaxis()
    # plt.show()

    return {i: np.stack(np.where(np.isclose(fimg, i * ker.sum())), axis=1) for i in cla_list}


# TODO@alfred(20220216): low pass filtering
def prelabel_center_extract(pre_map, overlap_thresh, nmsr, prob_thresh):
    bound_xy = non_max_suppression(pre_map[:, :, 1:].sum(axis=-1), overlap_thresh=overlap_thresh, max_boxes=1200,
                                   r=nmsr, prob_thresh=prob_thresh)
    coors = np.stack([(bound_xy[:, 1] + bound_xy[:, 3]) / 2,
                      (bound_xy[:, 0] + bound_xy[:, 2]) / 2], axis=1).astype(int)

    def find_cls(r, coors, pre_map):
        pre_cls_list = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                # TODO: which one?
                pre_cls_list.append(pre_map[coors[:, 0] + i, coors[:, 1] + j, :].argmax(axis=-1))
                # pre_cls_list.append(pre_map[coors[:, 0] + i, coors[:, 1] + j, 1:].argmax(axis=-1) + 1)
        pre_cls_list = np.stack(pre_cls_list, axis=-1)
        final_cls = []
        for cls_inx, count in [np.unique(arr, return_counts=True) for arr in pre_cls_list]:
            final_cls.append(cls_inx[count.argmax()])

        return np.array(final_cls)

    pre_cls = find_cls(r=2, coors=coors, pre_map=pre_map)

    return coors, pre_cls


def eval_pre(true_map, pre_map, r=6, overlap_thresh=0.3, nmsr=15, prob_thresh=0.65):
    '''

    :param true_map: shape = (b, h, w, class)
    :param pre_coors: shape = (b, h, w, class)
    :param r:
    :return:
    '''

    cls_list = np.unique(true_map[~np.isclose(true_map, 0)]).astype(int)

    pre_coors = []
    true_coors = []
    pre_clss = []
    for i in range(true_map.shape[0]):
        true_coors.append(truelabel_center_extract(true_map[i, :, :, 0], cls_list))
        coors, pre_cls = prelabel_center_extract(pre_map[i], overlap_thresh, nmsr, prob_thresh)
        pre_coors.append(coors)
        pre_clss.append(pre_cls)

    metrics = []
    for cls_j in cls_list:
        tp = 0
        fp = 0
        fn = 0
        for sub_i in range(true_map.shape[0]):
            pre_cls = pre_clss[sub_i]
            pre_coor = pre_coors[sub_i][pre_cls == cls_j]
            true_coor = true_coors[sub_i][cls_j]

            if pre_coor.shape[0] == 0:
                fn += len(true_coor)
            elif true_coor.shape[0] == 0:
                fp += len(pre_coor)
            else:
                eval_m = np.expand_dims(pre_coor, axis=1) - true_coor
                eval_v = np.sqrt((eval_m ** 2.).sum(axis=-1)).min(axis=1)

                tp += (eval_v <= r).sum()
                fp += (eval_v > r).sum()
                eval_m = np.expand_dims(true_coor, axis=1) - pre_coor
                eval_v = np.sqrt((eval_m ** 2.).sum(axis=-1)).min(axis=1)

                fn += (eval_v > r).sum()

                # eval_cls[pre_cls == cls_j] = eval_v < r
        # TODO: consider using different tp for precision and recall
        metrics.append([tp / (tp + fp + eps), tp / (tp + fn + eps)])
    metrics = np.array(metrics)
    p, r = metrics.mean(axis=0)
    return p, r


if __name__ == '__main__':
    from .config import Config
    from keras.optimizers import SGD
    from .model import data_prepare, SFCNnetwork, save_model_weights, det_model_compile, cls_model_compile, \
        joint_model_compile, generator_without_aug, generator_with_aug, generator_origin, tune_loss_weight

    CROP_SIZE = 64
    BATCH_SIZE = 1

    TRAIN_STEP_PER_EPOCH = 20
    NUM_TO_CROP, NUM_TO_AUG = 20, 10

    weights = tune_loss_weight()
    EPOCHS = Config.epoch
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'Cls_and_Det')

    data = data_prepare(data_dir, print_input_shape=True, print_image_shape=True)
    network = SFCNnetwork(l2_regularizer=weights[-1])
    optimizer = SGD(lr=Config.lr, momentum=0.9, decay=1e-6, nesterov=True)

    test_img, test_det, test_cls = data[6], data[7], data[8]

    model_weights_saver = save_model_weights('base', str(EPOCHS),
                                             dir_path=os.path.join(os.path.dirname(__file__), 'model_weights'))
    # Train Detection Branch

    prob_thresh = 0.5

    det_model = det_model_compile(nn=network, det_loss_weight=weights[0], optimizer=optimizer,
                                  softmax_trainable=False)
    det_model.load_weights(model_weights_saver[0])

    cls_model = cls_model_compile(nn=network, cls_loss_weight=weights[1], optimizer=optimizer,
                                  load_weights=model_weights_saver[0])
    cls_model.load_weights(model_weights_saver[1])

    joint_model = joint_model_compile(nn=network, det_loss_weight=weights[0], cls_loss_in_joint=weights[2],
                                      joint_loss_weight=weights[3], optimizer=optimizer,
                                      load_weights=model_weights_saver[1])
    # joint_model.load_weights(model_weights_saver[2])

    for im, la in generator_origin(test_img, test_det, test_cls,
                                   batch_size=test_img.shape[0], type='classification'):
        joint_pre = joint_model.predict(im)
        joint_pre = softmax(joint_pre, axis=-1)
        p, r = eval_pre(la, joint_pre, prob_thresh=prob_thresh)
        print('joint classification: precision: %.3f, recall: %.3f' % (p, r))
        break

    for im, la in generator_origin(test_img, test_det, test_cls,
                                   batch_size=test_img.shape[0], type='classification'):
        cls_pre = cls_model.predict(im)
        cls_pre = softmax(cls_pre, axis=-1)
        p, r = eval_pre(la, cls_pre, prob_thresh=prob_thresh)
        print('classification: precision: %.3f, recall: %.3f' % (p, r))
        break

    for im, la in generator_origin(test_img, test_det, test_cls,
                                   batch_size=test_img.shape[0], type='detection'):
        det_pre = det_model.predict(im)
        det_pre = softmax(det_pre, axis=-1)
        p, r = eval_pre(la, det_pre, prob_thresh=prob_thresh)
        print('detection: precision: %.3f, recall: %.3f' % (p, r))

        temp = np.zeros_like(det_pre)
        temp[:, :, :, 0] = joint_pre[:, :, :, 0]
        temp[:, :, :, 1] = joint_pre[:, :, :, 1:].sum(axis=-1)
        p, r = eval_pre(la, temp, prob_thresh=prob_thresh)
        print('detetion using joint model: precision: %.3f, recall: %.3f' % (p, r))
        break

    count = 0
    for im, la in generator_origin(test_img, test_det, test_cls,
                                   batch_size=BATCH_SIZE, type='detection'):
        im_in = im[0:1]
        la_in = la[0].squeeze()

        pre = det_model.predict(im_in)
        pre = softmax(pre, axis=-1)[0, :, :]

        joint_pre = joint_model.predict(im)
        joint_pre = softmax(joint_pre, axis=-1)
        pre[:, :, 0] = joint_pre[0, :, :, 0]
        pre[:, :, 1] = joint_pre[0, :, :, 1:].sum(axis=-1)
        prob_thresh = 0.7

        pre_coors, pre_cls = prelabel_center_extract(pre, overlap_thresh=0.3, nmsr=15, prob_thresh=prob_thresh)

        plt.figure(figsize=(12, 3))
        plt.subplot(141)
        plt.imshow(im_in[0].astype(np.uint8))

        plt.subplot(142)
        plt.imshow((pre[:, :, -1] * 255).astype(np.uint8))

        plt.subplot(143)
        # plt.imshow(im_in[0].astype(np.uint8), alpha=0.7)
        # plt.imshow((la_in * 255).astype(np.uint8), alpha=la_in)
        plt.imshow((pre[:, :, -1] * 255).astype(np.uint8))
        plt.scatter(pre_coors[:, 1], pre_coors[:, 0], s=15, alpha=0.4, c='tab:red')

        plt.subplot(144)
        plt.imshow((la_in * 255).astype(np.uint8), alpha=la_in)
        plt.scatter(pre_coors[:, 1], pre_coors[:, 0], s=15, alpha=0.4, c='tab:red')

        plt.show()

        count += 1
        if count > 4:
            break

    count = 0
    for im, la in generator_origin(test_img, test_det, test_cls,
                                   batch_size=BATCH_SIZE, type='classification'):

        im_in = im[0:1]
        la_in = la[0].squeeze()
        det_pre = det_model.predict(im_in)
        cls_pre = cls_model.predict(im_in)
        joint_pre = joint_model.predict(im_in)

        det_pre = softmax(det_pre, axis=-1)
        cls_pre = softmax(cls_pre, axis=-1)
        joint_pre = softmax(joint_pre, axis=-1)

        pre_coors, pre_cls = prelabel_center_extract(joint_pre[0], overlap_thresh=0.3, nmsr=15, prob_thresh=prob_thresh)
        # b = non_max_suppression(pre, overlap_thresh=0.3, max_boxes=1200, r=15, prob_thresh=0.65)

        plt.figure(figsize=(16, 8))
        plt.subplot(241)
        plt.imshow(im_in[0].astype(np.uint8))
        plt.title('ori image')

        plt.subplot(242)
        plt.matshow((det_pre[0, :, :, 1:].sum(axis=-1) * 255).astype(np.uint8), fignum=False)
        plt.title('det model')

        plt.subplot(243)
        # plt.imshow(im_in[0].astype(np.uint8), alpha=0.7)
        # plt.imshow((la_in * 255).astype(np.uint8), alpha=la_in)
        plt.matshow((cls_pre[0, :, :, 1:].sum(axis=-1) * 255).astype(np.uint8), fignum=False)
        # plt.scatter((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, s=15, alpha=0.4, c='tab:red')
        plt.title('cls model sum')

        plt.subplot(244)
        plt.matshow((joint_pre[0, :, :, 1:].sum(axis=-1) * 255).astype(np.uint8), fignum=False)
        # plt.scatter((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, s=15, alpha=0.4, c='tab:red')
        plt.title('joint model sum')

        plt.subplot(245)
        plt.matshow((la_in / 4 * 255).astype(np.uint8), alpha=(la_in > 0).astype(float), cmap='jet', fignum=False)
        plt.title('cls True label')

        plt.subplot(246)
        plt.matshow((cls_pre[0].argmax(axis=-1) / 4 * 255).astype(np.uint8), cmap='jet',
                    alpha=(cls_pre[0].argmax(axis=-1) > 0).astype(float), fignum=False)
        plt.title('cls model argmax')

        plt.subplot(247)
        jargmax = joint_pre[0].argmax(axis=-1)
        jargmax[0, 0] = 4  # align the range to true label
        plt.matshow((jargmax / 4 * 255).astype(np.uint8), cmap='jet', alpha=(jargmax > 0).astype(float),
                    fignum=False)
        plt.title('joint model argmax')

        plt.subplot(248)

        plt.matshow((jargmax / 4 * 255).astype(np.uint8), cmap='jet', alpha=(jargmax > 0).astype(float),
                    fignum=False)

        for i in np.unique(pre_cls):
            plt.scatter(pre_coors[pre_cls == i, 1], pre_coors[pre_cls == i, 0], s=15, alpha=0.4)
        plt.title('NMS')

        plt.show()

        count += 1
        if count > 4:
            break
