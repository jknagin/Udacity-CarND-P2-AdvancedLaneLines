import cv2
import numpy as np
from lib import utils
from typing import List


def find_chessboard_corners(img, nx: int, ny: int):
    ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)  # corners becomes imgpoints in calibrate_camera
    return ret, corners


def draw_chessboard_corners(img, nx: int, ny: int, corners, ret: bool):
    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    return img


def calibrate_camera(objpoints, imgpoints, gray_shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def calibrate_with_chessboard_images(calibration_images: List[str], nx: int, ny: int):
    objp = np.zeros((nx * ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    gray_shape = None

    for calibration_image in calibration_images:
        img = utils.read_image(calibration_image)
        gray = utils.grayscale(img)
        gray_shape = gray.shape
        ret, corners = find_chessboard_corners(gray, nx, ny)
        # print(calibration_image, ret)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            # img = draw_chessboard_corners(img, nx, ny, corners, ret)
            # plt.imshow(img)
            # plt.title(calibration_image)
            # plt.show()

    ret, mtx, dist, _, _ = calibrate_camera(objpoints, imgpoints, gray_shape)
    return ret, mtx, dist


def undistort(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted
