import cv2
import numpy as np


def get_perspective_transform(src, dst):
    # src: 4 point coordinates (X, Y) in original image
    # dst: 4 point coordinates (X, Y) in transformed image
    perspective_transform = cv2.getPerspectiveTransform(src, dst)
    return perspective_transform


def apply_perspective_transform(img, perspective_transform):
    transformed_image = cv2.warpPerspective(img, perspective_transform, (img.shape[1], img.shape[0]),
                                            flags=cv2.INTER_LINEAR)
    return transformed_image


def warp_to_roi(undistorted):
    img_shape = undistorted.shape

    # (lower_left, upper_left, upper_right, lower_right)
    # src_vertices = np.array([[img_shape[1] * 0.125, img_shape[0]],
    #                          [img_shape[1] * 0.465, img_shape[0] * 0.6],
    #                          [img_shape[1] * 0.550, img_shape[0] * 0.6],
    #                          [img_shape[1] * 0.950, img_shape[0]]], dtype=np.float32)
    src_vertices = np.array([[200, 720],
                             [599, 448],
                             [682, 448],
                             [1150, 720]], dtype=np.float32)

    # dst_vertices = np.array([[0.1 * img_shape[1], 0.9 * img_shape[0]],
    #                          [0.1 * img_shape[1], 0.1 * img_shape[0]],
    #                          [0.9 * img_shape[1], 0.1 * img_shape[1]],
    #                          [0.9 * img_shape[1], 0.9 * img_shape[0]]], dtype=np.float32)
    dst_vertices = np.array([[300, 720],
                             [300, 0],
                             [950, 0],
                             [950, 720]], dtype=np.float32)

    perspective_transform = get_perspective_transform(src_vertices, dst_vertices)
    img_transformed = apply_perspective_transform(undistorted, perspective_transform)
    return img_transformed
