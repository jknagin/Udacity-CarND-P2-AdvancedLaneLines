import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib import perspective

YM_PER_PIX = 30. / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

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

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

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


def fit_poly_helper(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calculate both polynomials using ploty, left_fit and right_fit
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    return left_fitx, right_fitx, ploty, left_fit, right_fit


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Perform polynomial fit
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly_helper(binary_warped.shape, leftx, lefty, rightx,
                                                                        righty)

    # Visualization
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, ploty, left_fitx, right_fitx, left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit):
    margin = 100

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values within the +/- margin of our polynomial function

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly_helper(binary_warped.shape, leftx, lefty, rightx,
                                                                        righty)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return result


def radius_of_curvature(fit, y_eval):
    a, b, _ = fit
    r = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / (np.abs(2 * a))
    return r


def angle_of_curvature(radius: float) -> float:
    # radius_of_curvature in meters
    # return angle of curvature in degrees
    return 100 / (2 * np.pi * radius) * 360


def measure_curvature(ploty, left_fit, right_fit, units: str):
    assert units in {"p", "m"}  # pixels or meters
    y_eval = np.max(ploty)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # Calculate new polynomial fit in meters
    left_fit_m = np.polyfit(ploty * YM_PER_PIX, left_fitx * XM_PER_PIX, 2)
    right_fit_m = np.polyfit(ploty * YM_PER_PIX, right_fitx * XM_PER_PIX, 2)

    left_radius, right_radius = 0, 1
    if units == "p":
        left_radius = radius_of_curvature(left_fit, y_eval)
        right_radius = radius_of_curvature(right_fit, y_eval)
    elif units == "m":
        left_radius = radius_of_curvature(left_fit_m, y_eval * YM_PER_PIX)
        right_radius = radius_of_curvature(right_fit_m, y_eval * YM_PER_PIX)

    return left_radius, right_radius


def vehicle_position_error(binary_warped, left_fit, right_fit):
    left_x_intercept = np.polyval(left_fit, binary_warped.shape[0])
    right_x_intercept = np.polyval(right_fit, binary_warped.shape[0])
    lane_centerline = (left_x_intercept + right_x_intercept) / 2  # lane centerline is between left and right lanes
    vehicle_pos = binary_warped.shape[1] / 2  # assume camera is mounted on center of car
    error = XM_PER_PIX * (
            vehicle_pos - lane_centerline)  # positioning error in meters, positive means car is to the right of center

    return error


def overlay_lane_area(undistorted, binary_warped, ploty, left_fitx, right_fitx, inverse_perspective_transform, left_radius, right_radius, left_angle, right_angle, error):
    # Create blank image to write to
    zero_channel = np.zeros_like(binary_warped).astype(np.uint8)
    lane_area_img = np.dstack((zero_channel, zero_channel, zero_channel))

    # Collect left and right lane points into a group of vertices defining the lane area
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill the inside of the vertices in the blank image
    cv2.fillPoly(lane_area_img, np.int_([pts]), (0, 255, 0))

    # Apply inverse perspective transform to the lane area
    lane_area_img_unwarped = perspective.apply_perspective_transform(lane_area_img, inverse_perspective_transform)

    # Calculate weighted sum of undistorted image overlaid with unwarped lane area
    undistorted_with_lane_area = cv2.addWeighted(undistorted, 1, lane_area_img_unwarped, 0.5, 0)

    # return undistorted_with_lane_area

    # Add text showing radius and angle of curvature and vehicle position
    font_family = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    font_size = 2
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(undistorted_with_lane_area, "Lane curvature radius: {} m".format(round(0.5*(left_radius + right_radius), 1)), (50, 50), font_family, font_size, font_color, font_thickness, line_type)
    cv2.putText(undistorted_with_lane_area, "Lane curvature angle: {} deg".format(round(0.5*(left_angle + right_angle), 1)), (50, 150), font_family, font_size, font_color, font_thickness, line_type)
    cv2.putText(undistorted_with_lane_area, "Vehicle is {} m {} of center".format(round(error, 2), "left" if error < 0 else "right"), (50, 250), font_family, font_size, font_color, font_thickness, line_type)

    return undistorted_with_lane_area
