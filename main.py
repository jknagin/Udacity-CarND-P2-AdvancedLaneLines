import glob
import matplotlib.pyplot as plt
import numpy as np

from lib import calibration, perspective, thresholding, curvature, utils
from moviepy.editor import VideoFileClip

calibration_images = glob.glob("camera_cal/calibration*.jpg")
nx = 9
ny = 6
_, MTX, DIST = calibration.calibrate_with_chessboard_images(calibration_images, nx, ny)


def pipeline_for_video(raw_image):
    return pipeline(raw_image, MTX, DIST)


def pipeline(raw_image, mtx, dist):
    # Undistort image
    undistorted = calibration.undistort(raw_image, mtx, dist)

    # Calculate binary image via sobel, rgb, and hsl thresholds
    sobel_kernel = 7
    abs_sobel_thresh = (20, 100)
    sobel_mag_thresh = (30, 100)

    red_thresh = (130, 255)
    green_thresh = (130, 255)
    blue_thresh = (190, 255)

    hue_thresh = (20, 90)
    lightness_thresh = (120, 200)
    saturation_thresh = (100, 255)

    sobel_thresh = (abs_sobel_thresh, sobel_mag_thresh)
    rgb_thresh = (red_thresh, green_thresh, blue_thresh)
    hls_thresh = (hue_thresh, lightness_thresh, saturation_thresh)
    binary_output = thresholding.binary_image(undistorted, sobel_kernel, sobel_thresh, rgb_thresh, hls_thresh)

    # utils.plot_two_images(undistorted, binary_output, "", "", (False, True))
    # utils.show()
    # return

    # Warp perspective of binary output to ROI
    src_vertices = np.array([[200, 720],
                             [599, 448],
                             [682, 448],
                             [1150, 720]], dtype=np.float32)
    dst_vertices = np.array([[300, 720],
                             [300, 0],
                             [950, 0],
                             [950, 720]], dtype=np.float32)

    binary_warped, perspective_transform, inverse_perspective_transform = perspective.warp_to_roi(binary_output, src_vertices, dst_vertices)

    # Perform polynomial fit on warped binary image
    ploty, left_fitx, right_fitx, left_fit, right_fit = curvature.fit_polynomial(binary_warped)

    # Measure the curvature of the polynomial fits
    left_radius_m, right_radius_m = curvature.measure_curvature(ploty, left_fit, right_fit, "m")

    # Calculate the vehicle positioning error relative to the center of the lane
    error = curvature.vehicle_position_error(binary_warped, left_fit, right_fit)

    # Display curvature and vehicle error info on the undistorted image
    lane_area_img = curvature.overlay_lane_area(undistorted, binary_warped, ploty, left_fitx, right_fitx,
                                                inverse_perspective_transform, left_radius_m, right_radius_m, error)

    return lane_area_img


def main():
    # Apply pipeline to each test image
    test_images = glob.glob("test_images/*.jpg")
    for test_image in test_images:
        # Read image
        raw_image = utils.read_image(test_image)

        # Apply pipeline using camera calibration parameters
        lane_area_img = pipeline(raw_image, MTX, DIST)

        # Visualize result of pipeline
        # plt.imshow(lane_area_img)
        # plt.title("{}".format(test_image))
        # plt.show()

    # # Apply pipeline to video
    # white_output = "project_video_output.mp4"
    # clip = VideoFileClip("project_video.mp4")
    # white_clip = clip.fl_image(pipeline_for_video)
    # white_clip.write_videofile(white_output, audio=False)


if __name__ == "__main__":
    main()
