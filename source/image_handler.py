# 图像处理 描框及获取位置

import cv2
import imutils
import numpy as np


def img_handler(_img: np.ndarray) -> tuple[np.ndarray, list[int or None, int or None]]:
    """图像处理函数 参数为原始图像 返回处理后的图像及目标物体的中心"""
    # 二值化图像
    gray_image = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY_INV)  # 反转二值化结果

    # 查找二值化后的轮廓并排序
    cnt = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_s = sorted(imutils.grab_contours(cnt), key=cv2.contourArea, reverse=True)

    # 根据离边框的位置,以及所需识别物体的像素块大小选择对应的轮廓
    target_bound = None
    for i in cnt_s:
        if 1500 > cv2.contourArea(i) > 10:
            bounding_box = cv2.boundingRect(i)
            if bounding_box[0] < 5 or bounding_box[1] < 5:
                continue
            if bounding_box[0] + bounding_box[2] > _img.shape[1] - 5:
                continue
            if bounding_box[1] + bounding_box[3] > _img.shape[0] - 5:
                continue
            if bounding_box[2] < 10 or bounding_box[3] < 30:
                continue
            if bounding_box[2] > 60 or bounding_box[3] > 200:
                continue
            if bounding_box[2] > 2 * bounding_box[3] + 20:  # 宽度不超过激光笔宽度+20像素
                continue
            if bounding_box[2] < bounding_box[3] * 1.1:
                target_bound = i
                break

    # 找到轮廓并储存相应的数值
    center_pos = [None, None]
    if target_bound is not None:
        bounding_box = cv2.boundingRect(target_bound)
        cv2.rectangle(_img, (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 0, 255), 2)

        # 根据轮廓查找中心
        center = cv2.moments(target_bound)
        if center["m00"] != 0:
            center_pos[0] = int(center["m10"] / center["m00"])
            center_pos[1] = int(center["m01"] / center["m00"])
            cv2.putText(_img, "Center:" + str(center_pos), (bounding_box[0] - 20, bounding_box[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return _img, center_pos


if __name__ == '__main__':
    pass
