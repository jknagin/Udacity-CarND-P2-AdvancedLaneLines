# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

A few points about this submission:
* This project submission should be run using the Jupyter notebook ```P2.ipynb``` in the top level directory of the submission repo. Hereafter, this notebook will be referred to as "the notebook."
* The notebook acts as a wrapper running code from the Python files under ```lib/```.
* The notebook saves the output images from each step of the image pipeline under ```test_images/```
* The notebook process the project video ```project_video.mp4``` and save it as ```project_video_output.mp4``` in the top-level directory of the submission repo.

[//]: # (Image References)

[image1]: ./output_images/pipeline_output/test_images/test1.jpg "Output of image pipeline on test1.jpg"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

The code for this step is contained under Section 0 of the notebook. The calibration images are collected together using ```glob``` and converted to grayscale. Then, ```cv2.findChessboardCorners()``` is called on each grayscale image to collect a sequence of ```imgpoints``` and ```objpoints```. This sequence of ```objpoints``` and ```imgpoints``` is passed to ```cv2.calibrateCamera()``` to compute the camera matrix and distortion coefficients.


### Pipeline (single images)

#### **1. Distortion Correction**

The code for this step is contained under Section 1 of the notebook. Using the camera matrix and distortion coefficients from Section 0, each of the calibration images is undistorted using ```cv2.undistort()```. The undistorted calibration images are saved to [```output_images/undistortion/camera_cal/```](./output_images/undistortion/camera_cal). The test images are also undisorted in the same manner. The undistorted test images are saved to [```output_images/undistortion/test_images/```](./output_images/undistortion/test_images).

#### **2. Thresholds**

The code for this step is contained under Section 2 of the notebook. Section 2 has four subsections labeled 2A - 2D:
* Section 2A uses Sobel gradient thresholds, including absolute Sobel X gradient and Sobel gradient magnitude thresholds, on the undistorted test images.
* Section 2B uses RGB thresholds on the undistorted test images, primarily to isolate white lane lines.
* Section 2C uses HLS thresholds on the undistorted test images, primarily to isolate yellow lane lines.
* Section 2D combines the thresholds in Sections 2A - 2C in a bitwise OR sense to produce binary images.

The binary images of each subsection 2A-2D are saved under [```output_images/thresholding/```](./output_images/thresholding).

#### **2A. Sobel Thresholds**

The code for this step is contained under Section 2A of the notebook. The purpose of these thresholds is to identify areas in the image with large, sudden changes. The Sobel gradients of each undistorted grayscaled image are computed using ```cv2.Sobel()```. The Sobel thresholds are the bitwise AND of the absolute value of the X gradient and the gradient magnitude (using their individual thresholds). 

#### **2B. RGB Thresholds**

The code for this step is contained under Section 2B of the notebook.  The purpose of these thresholds is to identify white lane lines. The RGB thresholds are the bitwise AND of the individual R, G, B channel thresholds.

#### **2C. HLS Thresholds**

The code for this step is contained under Section 2C of the notebook.  The purpose of these thresholds is to identify yellow lane lines. The HLS representation of each image is calculated using ```cv2.cvtColor()```. The HLS thresholds are the bitwise AND of the individual L and S channel thresholds, which are then bitwise OR'd with the individual H channel thresholds.

#### **2D. Combined Thresholds**

The code for this step is contained under Section 2D of the notebook. The purpose of combining the thresholds is to isolate sudden, large changes (via Sobel thresholds), white lane lines (via RGB thresholds), and yellow lane lines (via HLS thresholds).

#### **3. Perspective Transform**

The code for this step is contained under Section 3 of the notebook. The purpose of the perspective transform is to createa birds-eye view of a region of interest in each image where the lane is expected to be. This step implicitly assumes that the camera is already approximately in the center of a lane. This assumption holds true for each test image and the test video.

The perspective transform is computed by specifying hard-coded ```src_vertices``` and ```dst_vertices```, which are pixel coordinates within each image specifying the vertices of a quadrilateral region of interest. The vertices are then passed to ```cv2.getPerspectiveTransform()``` to combine the perspective transform and its inverse. The perspective transform is then applied to each binary image outputted from Section 2D via ```cv2.warpPerspective()``` to create binary warped ("birds-eye view") images.

The binary warped images are saved under [```output_images/perspective_warp/test_images/```](./output_images/perspective_warp/test_images).

#### **4. Perform Quadratic Polynomial Fitting on Each Lane Line**

The code for this step is contained under Section 4 of the notebook. the purpose of performing quadratic polynomial fitting on each lane line is to estimate the radius of curvature of the lane, and the vehicle positioning error relative to the center of the lane. The calculation of vehicle position error assumes that the camera is mounted in the center of the vehicle.

First, the left and right lane pixels coordinates of the binary warped images are identified using a sliding window of histograms approach. The left lane pixels are highlighted red, while the right lane pixels are highlighted blue. The histograms are calculated by summing the values of each pixel (1 or 0), column by column. The left and right topmost peaks of the histogram correspond with the locations of the left and right lane lines.

Second, the left and right lane pixel coordinates are used to calculate quadratic polynomial fits for the left and right lane lines. It is important to note that the polynomial fits are calculated using Y as the independent variable, instead of X, since the lane lines are likely to be close to vertical.

Third, the quadratic fits are applied to a sequence of Y values to produce fitted X values. The fitted X and Y values are overlaid on top of the binary warped image as yellow curves.

Fourth, the binary warped images with the left and right lanes colored, and with the quadratic fits shown in yellow, are saved under ```output_images/polynomial_fitting/test_images/```.

#### **5. Calculate Radii of Curvature and Vehicle Position Errors**

The code for this step is contained under Section 5 of the notebook. The purpose of this step is to use the quadratic fits of the lane lines from the previous step to compute the radius of curvature of the road and the vehicle position error with respect to the center of the lane.

First, the quadratic fits from the previous step are scaled from pixels to meters using the following conversion factors:

```python
YM_PER_PIX = 30. / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension
```

Second, the radius of curvature is computed using the following equation:
```python
r = (1 + (2 * a * y_eval + b) ** 2) ** 1.5 / (np.abs(2 * a))
```
where ```a``` and ```b``` are the second and first order term coefficients of the quadratic fit, respectively, and ```y_eval``` is taken to be the bottom of the image so as to compute the curvature of the road where the vehicle is currently located.


#### **6. Display Radius of Curvature and Vehicle Position Error**

The code for this step is contained under Section 6 of the notebook. The purpose of this step is to overlay the radius of curvature and vehicle position error information from the previous step onto the undistorted image. This represents the final output of the image pipeline.

The images are saved under [```output_images/pipeline_output/test_images```](./output_images/pipeline_output/test_images). Below is the output of the image pipeline on [```test_images/test1.jpg```](./test_images/test1.jpg):

![image1]
---

### Pipeline (video)

The code for this step is contained under Section 7 of the notebook. The purpose of this section is to use the image pipeline to process a video of a car driving on a highway frame by frame and highlight the lane for the duration of the video.  Here's a [link to my video result](./project_video_output.mp4).

---

### Discussion

Using the RGB and HLS color thresholds to separately identify white and yellow lanes, respectively, seemed to work quite well. The white lanes were well-suited to identification with RGB thresholds because the RGB value of pure white is (255, 255, 255). The yellow lanes were well-suited to identification with HLS thresholds because the H value of yellow is around 40-60, regardless of lightness or saturation.

The most crucial part of the image pipeline is probably when the undistorted images are converted to binary images using the various color and Sobel gradient thresholds (Section 2). This will fail if, for example, the lane markings are not highly saturated, which can happen due to wear and tear of the road.

The image pipeline will also fail if the vehicle is not already well-centered within the lane, since the perspective transform assumes a static region of interest roughly centered around the center of the view of the camera. The vehicle will not be well-centered during a lane change.

The pipeline could be improved by, for example, implementing a search from prior during the sliding window of histograms step to improve the efficiency of the computation.