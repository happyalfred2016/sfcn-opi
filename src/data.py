import numpy as np
from numpy.random import randint
from imgaug import augmenters as iaa
import os
import cv2


def _image_normalization(image):
    # TODO@alfred: this will affect the downstream augmentation which are only tested on uint8
    img = image
    assert img.dtype == np.uint8

    # img = image / 255.
    # img -= np.mean(img, keepdims=True)
    # img /= (np.std(img, keepdims=True) + 1e-7)
    return img


def crop_image_batch(image, masks=None, if_mask=True, if_det=True, if_cls=True,
                     origin_shape=(500, 500), desired_shape=(64, 64)):
    assert image.ndim == 4
    ori_width, ori_height = origin_shape[0], origin_shape[1]
    des_width, des_height = desired_shape[0], desired_shape[1]

    max_x = ori_width - des_width
    max_y = ori_height - des_height
    ran_x = np.random.randint(0, max_x)
    ran_y = np.random.randint(0, max_y)
    cropped_x = ran_x + des_width
    cropped_y = ran_y + des_height
    cropped_img = image[:, ran_x:cropped_x, ran_y:cropped_y]
    if if_mask and masks is not None:
        if if_det and if_cls:
            det_mask = masks[0]
            cls_mask = masks[1]
            cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
            cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y]
            return cropped_img, cropped_det_mask, cropped_cls_mask, ran_x, ran_y
        elif if_det and not if_cls:
            det_mask = masks
            cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
            return cropped_img, cropped_det_mask, ran_x, ran_y
        elif if_cls and not if_det:
            cls_mask = masks
            cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y, :]
            return cropped_img, cropped_cls_mask, ran_x, ran_y
    else:
        return cropped_img, ran_x, ran_y


def crop_image(image, masks=None, if_mask=True, if_det=True, if_cls=True,
               origin_shape=(500, 500), desired_shape=(64, 64)):
    assert image.ndim == 3
    ori_width, ori_height = origin_shape[0], origin_shape[1]
    des_width, des_height = desired_shape[0], desired_shape[1]

    max_x = ori_width - des_width
    max_y = ori_height - des_height
    ran_x = randint(0, max_x)
    ran_y = randint(0, max_y)
    cropped_x = ran_x + des_width
    cropped_y = ran_y + des_height
    cropped_img = image[ran_x:cropped_x, ran_y:cropped_y]
    if if_mask and masks is not None:
        if if_det and if_cls:
            det_mask = masks[0]
            cls_mask = masks[1]
            cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
            cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
            return cropped_img, cropped_det_mask, cropped_cls_mask
        elif if_det and not if_cls:
            det_mask = masks[0]
            cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
            return cropped_img, cropped_det_mask
        elif if_cls and not if_det:
            cls_mask = masks[0]
            cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
            return cropped_img, cropped_cls_mask
    else:
        return cropped_img


def crop_on_fly(img, det_mask, cls_mask, crop_size):
    """
    Crop image randomly on each training batch
    """
    cropped_img, cropped_det_mask, cropped_cls_mask = crop_image_batch(img, [det_mask, cls_mask],
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

                    # iaa.Affine(shear=(-16, 16)),
                    # iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                    # iaa.PerspectiveTransform(scale=(0.01, 0.1))

                    iaa.Affine(shear=(-8, 8)),
                    iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                    # iaa.PerspectiveTransform(scale=(0.01, 0.1))

                ]))
        ])

        seq_to_deterministic = seq.to_deterministic()
        aug_img, aug_det_mask = seq_to_deterministic(images=image, segmentation_maps=masks[0])
        _aug_img, aug_cls_mask = seq_to_deterministic(images=image, segmentation_maps=masks[1])

        assert np.isclose(_aug_img, aug_img).all()
        return aug_img, aug_det_mask, aug_cls_mask

    aug_image, aug_det_mask, aug_cls_mask = image_basic_augmentation(image=img, masks=[det_mask, cls_mask])
    return aug_image, aug_det_mask, aug_cls_mask


def load_data(data_path, type, det=True, cls=True, reshape_size=None):
    path = os.path.join(data_path, type)  # Cls_and_Det/train
    imgs, det_masks, cls_masks = [], [], []
    for i, file in enumerate(os.listdir(path)):
        for j, img_file in enumerate(os.listdir(os.path.join(path, file))):
            if '.bmp' in img_file:
                img_path = os.path.join(path, file, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if reshape_size is not None:
                    img = cv2.resize(img, reshape_size)
                img = _image_normalization(img)
                imgs.append(img)
            if 'mask' in img_file and det == True:
                det_mask_path = os.path.join(path, file, img_file, 'detection', 'det_%s.png' % file)
                # det_mask = skimage.io.imread(det_mask_path, True).astype(np.bool)
                det_mask = cv2.imread(det_mask_path, 0)
                if reshape_size is not None:
                    det_mask = cv2.resize(det_mask, reshape_size)
                # TODO@alfred
                det_mask[det_mask > 0] = 1
                det_masks.append(det_mask)
            if 'mask' in img_file and cls == True:
                cls_mask_path = os.path.join(path, file, img_file, 'classification', 'cls_%s.png' % file)
                cls_mask = cv2.imread(cls_mask_path, 0)
                if reshape_size != None:
                    cls_mask = cv2.resize(cls_mask, reshape_size)

                # TODO@alfred
                cls_mask[cls_mask == 56] = 1
                cls_mask[cls_mask == 106] = 2
                cls_mask[cls_mask == 156] = 3
                cls_mask[cls_mask == 206] = 4
                cls_masks.append(cls_mask)
    return np.array(imgs), np.array(det_masks), np.array(cls_masks)


def data_prepare(data_dir, print_image_shape=False, print_input_shape=False):
    """
    prepare data for model.
    :param print_image_shape: print image shape if set true.
    :param print_input_shape: print input shape(after categorize) if set true
    :return: list of input to model
    """

    def reshape_mask(origin, cate, num_class):
        return cate.reshape((origin.shape[0], origin.shape[1], origin.shape[2], num_class))

    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=data_dir, type='train')
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=data_dir, type='validation')
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=data_dir, type='test')

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


def generator_with_aug(features, det_labels, cls_labels, batch_size, crop_size,
                       type, crop_num=15, aug_num=10):
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
                          type, crop_num=25):
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
    batch_det_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, det_labels.shape[-1]))
    batch_cls_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, cls_labels.shape[-1]))
    while True:
        counter = 0
        for i in range(batch_size):
            index = np.random.choice(features.shape[0], 1)
            for j in range(crop_num):
                feature, det_label, cls_label = crop_on_fly(features[index], det_labels[index],
                                                            cls_labels[index], crop_size=crop_size)
                batch_features[counter] = feature
                batch_det_labels[counter] = det_label
                batch_cls_labels[counter] = cls_label
                counter += 1
        if type == 'detection':
            yield batch_features, batch_det_labels
        elif type == 'classification' or type == 'joint':
            yield batch_features, batch_cls_labels


def generator_origin(features, det_labels, cls_labels, batch_size, type):
    img_h, img_w = features.shape[1], features.shape[2]
    batch_features = np.zeros((batch_size, img_h, img_w, 3))
    batch_det_labels = np.zeros((batch_size, img_h, img_w, det_labels.shape[-1]))
    batch_cls_labels = np.zeros((batch_size, img_h, img_w, cls_labels.shape[-1]))

    counter = 0
    while True:

        nbat = counter % int(np.ceil(features.shape[0] / batch_size))
        counter += 1
        for index in range(nbat * batch_size, min((nbat + 1) * batch_size, features.shape[0])):
            batch_features[index % batch_size] = features[index]
            batch_det_labels[index % batch_size] = det_labels[index]
            batch_cls_labels[index % batch_size] = cls_labels[index]

        if type == 'detection':
            yield batch_features, batch_det_labels
        elif type == 'classification' or type == 'joint':
            yield batch_features, batch_cls_labels
