import os
import cv2
from numpy.random import randint
import numpy as np
from imgaug import augmenters as iaa
from .util import check_directory, check_cv2_imwrite
from .data_manager import DataManager as dm


def crop_image_parts(image, det_mask, cls_mask, origin_shape=(512, 512)):
    assert image.ndim == 3
    ori_width, ori_height = origin_shape[0], origin_shape[1]
    des_width, des_height = 256, 256
    assert des_width == des_height

    cropped_img1 = image[0: des_width, 0: des_height, :]  # 1, 3
    cropped_img2 = image[des_width: ori_width, 0: des_height, :]  # 2, 4
    cropped_img3 = image[0: des_width, des_height: ori_height, :]
    cropped_img4 = image[des_width: ori_width, des_height: ori_height, :]

    cropped_det_mask1 = det_mask[0: des_width, 0: des_height, :]  # 1, 3
    cropped_det_mask2 = det_mask[des_width: ori_width, 0: des_height, :]  # 2, 4
    cropped_det_mask3 = det_mask[0: des_width, des_height: ori_height, :]
    cropped_det_mask4 = det_mask[des_width: ori_width, des_height: ori_height, :]

    cropped_cls_mask1 = cls_mask[0: des_width, 0: des_height, :]  # 1, 3
    cropped_cls_mask2 = cls_mask[des_width: ori_width, 0: des_height, :]  # 2, 4
    cropped_cls_mask3 = cls_mask[0: des_width, des_height: ori_height, :]
    cropped_cls_mask4 = cls_mask[des_width: ori_width, des_height: ori_height, :]

    return [cropped_img1, cropped_img2, cropped_img3, cropped_img4,
            cropped_det_mask1, cropped_det_mask2, cropped_det_mask3, cropped_det_mask4,
            cropped_cls_mask1, cropped_cls_mask2, cropped_cls_mask3, cropped_cls_mask4,
            cropped_img1, cropped_img2, cropped_img3, cropped_img4]


def batch_crop_image_parts(ori_set, target_set):
    for file in os.listdir(ori_set):
        print(file)
        image_file = os.path.join(ori_set, str(file), str(file) + '.bmp')
        det_mask_file = os.path.join(ori_set, str(file), str(file) + '_detection.bmp')
        cls_mask_file = os.path.join(ori_set, str(file), str(file) + '_classification.bmp')
        image = cv2.imread(image_file)
        image = cv2.resize(image, (512, 512))
        det_mask = cv2.imread(det_mask_file)
        det_mask = cv2.resize(det_mask, (512, 512))
        cls_mask = cv2.imread(cls_mask_file)
        cls_mask = cv2.resize(cls_mask, (512, 512))
        crop_list = crop_image_parts(image, det_mask, cls_mask)

        list_file_create = [os.path.join(target_set, str(file) + '_1'),
                            os.path.join(target_set, str(file) + '_2'),
                            os.path.join(target_set, str(file) + '_3'),
                            os.path.join(target_set, str(file) + '_4')]
        check_directory(list_file_create)
        list_img_create = [os.path.join(target_set, str(file) + '_1', str(file) + '_1.bmp'),
                           os.path.join(target_set, str(file) + '_2', str(file) + '_2.bmp'),
                           os.path.join(target_set, str(file) + '_3', str(file) + '_3.bmp'),
                           os.path.join(target_set, str(file) + '_4', str(file) + '_4.bmp'),
                           os.path.join(target_set, str(file) + '_1', str(file) + '_1_detection.bmp'),
                           os.path.join(target_set, str(file) + '_2', str(file) + '_2_detection.bmp'),
                           os.path.join(target_set, str(file) + '_3', str(file) + '_3_detection.bmp'),
                           os.path.join(target_set, str(file) + '_4', str(file) + '_4_detection.bmp'),
                           os.path.join(target_set, str(file) + '_1', str(file) + '_1_classification.bmp'),
                           os.path.join(target_set, str(file) + '_2', str(file) + '_2_classification.bmp'),
                           os.path.join(target_set, str(file) + '_3', str(file) + '_3_classification.bmp'),
                           os.path.join(target_set, str(file) + '_4', str(file) + '_4_classification.bmp'),
                           os.path.join(target_set, str(file) + '_1', str(file) + '_1_original.bmp'),
                           os.path.join(target_set, str(file) + '_2', str(file) + '_2_original.bmp'),
                           os.path.join(target_set, str(file) + '_3', str(file) + '_3_original.bmp'),
                           os.path.join(target_set, str(file) + '_4', str(file) + '_4_original.bmp')]

        for order, img in enumerate(crop_list):
            check_cv2_imwrite(list_img_create[order], img)
        # check_directory(list_file_create)
        # cv2.imwrite


class ImageCropping:
    def __init__(self, data_path=None, old_filename=None, new_filename=None):
        self.data_path = data_path
        self.old_filename = '{}/{}'.format(data_path, old_filename)
        self.new_filename = '{}/{}'.format(data_path, new_filename)
        dm.check_directory(self.new_filename)
        dm.initialize_train_test_folder(self.new_filename)

    @staticmethod
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
                return cropped_img, cropped_det_mask, cropped_cls_mask
            elif if_det and not if_cls:
                det_mask = masks
                cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
                return cropped_img, cropped_det_mask
            elif if_cls and not if_det:
                cls_mask = masks
                cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y, :]
                return cropped_img, cropped_cls_mask
        else:
            return cropped_img

    @staticmethod
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


if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    if ROOT_DIR.endswith('src'):
        ROOT_DIR = os.path.dirname(ROOT_DIR)

    OLD_DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'Cls_and_Det')
    TRAIN_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'train')
    TEST_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'test')
    VALID_OLD_DATA_DIR = os.path.join(OLD_DATA_DIR, 'validation')
    TARGET_DATA_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det')
    TRAIN_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'train')
    TEST_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'test')
    VALID_TARGET_DATA_DIR = os.path.join(TARGET_DATA_DIR, 'validation')

    batch_crop_image_parts(TRAIN_OLD_DATA_DIR, TRAIN_TARGET_DATA_DIR)
    batch_crop_image_parts(TEST_OLD_DATA_DIR, TEST_TARGET_DATA_DIR)
    batch_crop_image_parts(VALID_OLD_DATA_DIR, VALID_TARGET_DATA_DIR)
