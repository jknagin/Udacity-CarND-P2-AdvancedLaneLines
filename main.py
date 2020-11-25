import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

from lib import calibration, perspective, utils, thresholding


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Hyperparameters
    # Choose the number of sliding windows
    num_windows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    min_pix = 50

    # Set height of windows - based on num_windows above and image shape
    window_height = np.int(binary_warped.shape[0] // num_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Current positions to be updated later for each window in num_windows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > min_pix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_pix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > min_pix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Visualization #
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img


def main():
    calibration_images = glob.glob("camera_cal/calibration*.jpg")
    nx = 9
    ny = 6

    ret, mtx, dist = calibration.calibrate_with_chessboard_images(calibration_images, nx, ny)

    test_images = glob.glob("test_images/*.jpg")
    straight_lines = glob.glob("test_images/straight_lines*.jpg")
    for test_image in test_images:
        print(test_image)
        img = utils.read_image(test_image)
        undistorted = calibration.undistort(img, mtx, dist)

        # Calculate binary matrices via thresholds
        sobel_kernel = 3
        abs_sobel_thresh = (0, 255)
        sobel_mag_thresh = (0, 255)

        red_thresh = (0, 255)
        green_thresh = (0, 255)
        blue_thresh = (0, 255)

        hue_thresh = (0, 255)
        lightness_thresh = (0, 255)
        saturation_thresh = (0, 255)

        sobel_thresh = (abs_sobel_thresh, sobel_mag_thresh)
        rgb_thresh = (red_thresh, green_thresh, blue_thresh)
        hls_thresh = (hue_thresh, lightness_thresh, saturation_thresh)
        # binary_output = thresholding.binary_image(undistorted, sobel_kernel, sobel_thresh, rgb_thresh, hls_thresh)
        binary_output = thresholding.blue_threshold(undistorted, blue_thresh)
        print(binary_output)

        utils.plot_two_images(undistorted, binary_output, "Undistorted", "Binary", binary=(False, True))
        utils.show()
        continue

        # Warp perspective of binary output to ROI
        binary_warped = perspective.warp_to_roi(binary_output)

        # Perform polynomial fit on warped binary image
        out_img = fit_polynomial(binary_warped)
        plt.imshow(out_img)
        plt.title("Sliding Windows: {}".format(test_image))
        plt.figure()
        plt.imshow(perspective.warp_to_roi(undistorted))
        plt.show()

        # Observe pipeline component effect
        # show_two_images(undistorted, out_img, "undistorted", "poly fit")
        # plt.show()


if __name__ == "__main__":
    main()
