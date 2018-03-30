import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import cv2
import random

import skimage
import imageio

def grayscale(image):
        """把image转换成8位图像（只有黑白色）
        """
        gray = skimage.color.rgb2gray(image.astype('uint8'))
        image = skimage.color.gray2rgb(gray)
        # image *= 255.   # 这里乘以255来把图片的像素转回到0~255，rgb2gray会把图片的像素值压到0到1之间
        return image


def get_mask_from_image(image, masks_dir):
    """对某个图片，和其关联的一堆Mask图片，返回所有的Mask举证
    注意这里会处理重叠，类似剪纸
    """
    mask_image_names = os.listdir(masks_dir)
    count = len(mask_image_names)
    masks = []

    for i, file_name in enumerate(mask_image_names):
        mask_iamge_path = os.path.join(masks_dir, file_name)
        mask_image = imageio.imread(mask_iamge_path)
        if np.sum(mask_image) == 0:
            print('invalid mask')
            continue
        mask_image = mask_image.astype('float32')/255.
        masks.append(mask_image)

    masks = np.asarray(masks)
    masks[masks > 0.] = 1.
    masks = np.transpose(masks, (1,2,0))

    occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
    count = masks.shape[2]
    for i in range(count-2, -1, -1):
        masks[:, :, i] = masks[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))

    return masks


def apply_mask_to_image(image, mask, rgba):
    """在对应的图片上添加mask
    """
    new_image = image.copy()
    rgba = np.asarray(rgba)
    ids = (mask==1)
    new_image[ids] = rgba
    return new_image


def extract_bboxes(masks):
    """从masks上获取bounding box，支持batch
    mask: [batch_size, height, width].  mask矩阵由1 或 0组成
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(masks.shape[0]):
        m = masks[0]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


def random_flip(image, masks):
    """随机水平翻转
    """
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        ret_masks = []
        for i in range(masks.shape[2]):
            mask = masks[:, :, i]
            ret_masks.append( cv2.flip(mask, 1) )
        masks = np.asarray(ret_masks)
        masks = np.transpose(masks, (1, 2, 0))
    return image, masks


def random_crop(image, masks, coe=0.7):
    """随机剪裁，大小为原图的coe倍
    """
    if np.random.random() < 0.5:
        if coe >= 1:
            return image, mask

        h, w = image.shape[:2]
        nh, nw = ( int(h * coe), int(w * coe) )
        random_x = random.choice(  range( 0, w - nw )   )
        random_y = random.choice( range( 0, h - nh )  )

        image = image[random_y:(random_y+nh), random_x:(random_x+nw), :]

        ret_masks = []
        for i in range(masks.shape[2]):
            mask = masks[:,:,i]
            mask = mask[random_y:(random_y+nh), random_x:(random_x + nw)]
            ret_masks.append(mask)
        masks = np.transpose(np.asarray(ret_masks), (1, 2, 0))

    return image, masks


def random_rotate(image, masks, degree_limit=90):
    """随机旋转
    """
    if np.random.random() < 0.5:
        h, w = image.shape[:2]
        degree = random.choice(range(degree_limit))

        M = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)

        image = cv2.warpAffine(image,M,(w, h))

        ret_masks = []
        for i in range(masks.shape[2]):
            mask = masks[:,:,i]
            mask = cv2.warpAffine(mask, M, (w, h))
            ret_masks.append(mask)
        masks = np.transpose(np.asarray(ret_masks), (1, 2, 0))

    return image, masks


# 亮度，饱和度，对比度随机调整
def random_brightness_transform(image, limit=[0.5,1.5]):
    if np.random.random() < 0.5:
        image = image.copy()
        alpha = np.random.uniform(limit[0], limit[1])
        image = alpha*image
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
