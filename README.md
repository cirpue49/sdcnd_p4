##Advanced Lane Finding Project

---


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_img.png "Road Transformed"
[image3]: ./examples/binary_img.png "Binary Example"
[image4]: ./examples/before_warp.png "Warp Example 1"
[image5]: ./examples/warped_color.png "Warp Example 2"
[image6]: ./examples/binary_warped.png "Warp Example 3"
[image7]: ./examples/find_lane.png "Two lanes"
[image8]: ./examples/right_left.png "Right and Left lanes"
[image9]: ./examples/poly_fit.png " "Poly fit"
[image10]: ./examples/result1.png " "Result 1"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./project_report.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how I used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (`Creating binary image` section in `project_report.ipynb`).  Here's an example of my output for this step. 
```
x_binary = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(50, 150))
y_binary = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(50, 150))
mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(50, 150))
hls_binary = hls_select(image, channel = 2, thresh=(120, 200))
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
combined = np.zeros_like(mag_binary)
combined[ ((hls_binary == 1) & (dir_binary==1))|( (mag_binary == 1) & (x_binary==1) & (y_binary ==1))] = 1
```

![alt text][image3]

####3. Describe how I performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in `Transforming perspective` section in `project report.ipynb`.  The `warp()` function takes as inputs an image (`img`). I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image5]

#####Applying binary threshhold

![alt text][image6]

####4. Describe how I identified lane-line pixels and fit their positions with a polynomial.

The code for my fitting with a polynomial is in `Fitting with a polynomial` section in `project report.ipynb`. 

First, I found right and left lane.

![alt text][image7]
![alt text][image8]

And, then fitting with polynomials.

![alt text][image9]

####5. Describe how I calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

```
def measuring_curv(l_x, l_y, r_x, r_y):
    l_max = 720
    r_max = 720
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(l_y*ym_per_pix, l_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(r_y*ym_per_pix, r_x*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*l_max + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*r_max + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    average_curv = (left_curverad + right_curverad)/2
    return average_curv


def get_text_info(img, l_x, l_y, r_x, r_y, l_lane_pix, r_lane_pix):
    # meters from center
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    screen_middel_pixel = img.shape[1]/2
    car_middle_pixel = int((r_lane_pix + l_lane_pix)/2)
    screen_off_center = screen_middel_pixel-car_middle_pixel
    meters_off_center = round(xm_per_pix * screen_off_center, 2)
    curv_in_meters = int(measuring_curv(l_x, l_y, r_x, r_y))
    return meters_off_center, curv_in_meters
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image10]

---

###Pipeline (video)

####1. Provide a link to your final video output.  

[Project Video](https://www.youtube.com/watch?v=vySgXdDJlrs&feature=youtu.be)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vySgXdDJlrs/0.jpg)](https://www.youtube.com/watch?v=vySgXdDJlrs)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This is traditional way of implementing lane detection. For the future work, I want to implement lane detection by using deep learning.

# sdcnd_p4
