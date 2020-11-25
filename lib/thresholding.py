import cv2
import numpy as np
from typing import Tuple

from lib import utils


def abs_sobel_threshold(gray, orient='x', sobel_kernel=3, thresh: Tuple[int, int] = (0, 255)):
    assert orient in ['x', 'y']
    direction = (1, 0) if orient == 'x' else (0, 1)
    gray = cv2.GaussianBlur(gray, (sobel_kernel, sobel_kernel), 0)
    abs_derivative = np.absolute(cv2.Sobel(gray, cv2.CV_64F, *direction, ksize=sobel_kernel))
    abs_derivative_scaled = np.uint8(abs_derivative * 255 / np.max(abs_derivative))
    return _check_threshold_helper(abs_derivative_scaled, thresh)


def sobel_magnitude_threshold(gray, sobel_kernel: int = 3, thresh: Tuple[int, int] = (0, 255)):
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude_scaled = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    return _check_threshold_helper(gradient_magnitude_scaled, thresh)


def red_threshold(undistorted, thresh: Tuple[int, int] = (0, 255)):
    return _check_threshold_helper(undistorted[:, :, 0], thresh)


def green_threshold(undistorted, thresh: Tuple[int, int] = (0, 255)):
    return _check_threshold_helper(undistorted[:, :, 1], thresh)


def blue_threshold(undistorted, thresh: Tuple[int, int] = (0, 255)):
    return _check_threshold_helper(undistorted[:, :, 2], thresh)


def hue_threshold(undistorted, thresh: Tuple[int, int] = (0, 255)):
    undistorted_hls = utils.hls(undistorted)
    return _check_threshold_helper(undistorted_hls[:, :, 0], thresh, left_inclusive=False)


def lightness_threshold(undistorted, thresh: Tuple[int, int] = (0, 255)):
    undistorted_hls = utils.hls(undistorted)
    return _check_threshold_helper(undistorted_hls[:, :, 0], thresh, left_inclusive=False)


def saturation_threshold(undistorted, thresh: Tuple[int, int] = (0, 255)):
    undistorted_hls = utils.hls(undistorted)
    return _check_threshold_helper(undistorted_hls[:, :, 2], thresh, left_inclusive=False)


def sobel_thresholds(gray, sobel_kernel: int = 3, thresholds: Tuple[Tuple[int]] = ((0, 255), (0, 255))):
    assert len(thresholds) == 2
    abs_sobel_thresh, sobel_mag_thresh = thresholds
    abs_sobel_binary = abs_sobel_threshold(gray, 'x', sobel_kernel=sobel_kernel, thresh=abs_sobel_thresh)
    sobel_mag_binary = sobel_magnitude_threshold(gray, sobel_kernel=sobel_kernel, thresh=sobel_mag_thresh)

    return abs_sobel_binary & sobel_mag_binary


def rgb_threshold(undistorted, thresholds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 255), (0, 255), (0, 255))):
    assert len(thresholds) == 3
    r_thresh, g_thresh, b_thresh = thresholds
    r_binary = red_threshold(undistorted, r_thresh)
    g_binary = green_threshold(undistorted, g_thresh)
    b_binary = blue_threshold(undistorted, b_thresh)

    return r_binary & g_binary & b_binary


def hls_threshold(undistorted, thresholds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 255), (0, 255), (0, 255))):
    assert len(thresholds) == 3
    h_thresh, l_thresh, s_thresh = thresholds
    h_binary = hue_threshold(undistorted, h_thresh)
    l_binary = lightness_threshold(undistorted, l_thresh)
    s_binary = saturation_threshold(undistorted, s_thresh)

    return h_binary & l_binary & s_binary


def binary_image(undistorted, sobel_kernel: int, sobel_thresh, rgb_thresh, hls_thresh):
    gray = utils.grayscale(undistorted)
    sobel_binary = sobel_thresholds(gray, sobel_kernel=sobel_kernel, thresholds=sobel_thresh)
    rgb_binary = rgb_threshold(undistorted, rgb_thresh)
    hls_binary = hls_threshold(undistorted, hls_thresh)

    return sobel_binary | rgb_binary | hls_binary


def _check_threshold_helper(channel, thresh: Tuple[int, int] = (0, 255), left_inclusive: bool = True):
    binary_output = np.zeros_like(channel)
    if left_inclusive:
        binary_output[(thresh[0] <= channel) & (channel <= thresh[1])] = 1
    else:
        binary_output[(thresh[0] < channel) & (channel <= thresh[1])] = 1
    return binary_output
