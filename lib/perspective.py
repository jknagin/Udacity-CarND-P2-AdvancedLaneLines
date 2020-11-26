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


def warp_to_roi(undistorted, src_vertices, dst_vertices):
    perspective_transform = get_perspective_transform(src_vertices, dst_vertices)
    inverse_perspective_transform = get_perspective_transform(dst_vertices, src_vertices)
    img_transformed = apply_perspective_transform(undistorted, perspective_transform)
    return img_transformed, perspective_transform, inverse_perspective_transform
