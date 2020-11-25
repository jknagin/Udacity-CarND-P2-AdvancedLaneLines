import glob
import matplotlib.pyplot as plt

from lib import calibration, perspective, utils, thresholding, curvature


def main():
    # Compute camera calibration parameters using camera_cal directory
    calibration_images = glob.glob("camera_cal/calibration*.jpg")
    nx = 9
    ny = 6
    ret, mtx, dist = calibration.calibrate_with_chessboard_images(calibration_images, nx, ny)

    test_images = glob.glob("test_images/*.jpg")
    # for test_image in ["test_images/test1.jpg", "test_images/test4.jpg"]:
    for test_image in test_images:
        # Read image
        img = utils.read_image(test_image)

        # Undistort image
        undistorted = calibration.undistort(img, mtx, dist)

        # Calculate binary image via sobel, rgb, and hsl thresholds
        sobel_kernel = 7
        abs_sobel_thresh = (20, 100)
        sobel_mag_thresh = (30, 100)

        red_thresh = (130, 255)
        green_thresh = (130, 255)
        blue_thresh = (190, 255)

        hue_thresh = (20, 90)
        lightness_thresh = (0, 255)
        saturation_thresh = (0, 255)

        sobel_thresh = (abs_sobel_thresh, sobel_mag_thresh)
        rgb_thresh = (red_thresh, green_thresh, blue_thresh)
        hls_thresh = (hue_thresh, lightness_thresh, saturation_thresh)
        binary_output = thresholding.binary_image(undistorted, sobel_kernel, sobel_thresh, rgb_thresh, hls_thresh)

        # Warp perspective of binary output to ROI
        binary_warped = perspective.warp_to_roi(binary_output)

        # Perform polynomial fit on warped binary image
        out_img, ploty, left_fit, right_fit = curvature.fit_polynomial(binary_warped)
        left_radius_m, right_radius_m = curvature.measure_curvature(ploty, left_fit, right_fit, "m")

        print(test_image, left_radius_m, right_radius_m)

        plt.imshow(out_img)
        plt.title("Sliding Windows: {}".format(test_image))
        # plt.figure()
        # plt.imshow(perspective.warp_to_roi(undistorted))
        plt.show()


if __name__ == "__main__":
    main()
