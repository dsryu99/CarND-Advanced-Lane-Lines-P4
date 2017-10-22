# 1. compute the camera calibration matrix and distortion coefficients
# 2. undistort each new frame
# 3. apply thresholds to create a binary image
# 4. apply a perspective transform
# 5. detect lane lines
# 6. determine the lane curvature

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

isFirst = True
prev_src = np.empty((0))
vertices = np.empty((0))
l_fit = np.empty((0))
r_fit = np.empty((0))

def cal_undistort(img, mtx, dist):
#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #undist = np.copy(img)  # Delete this line
    return undist

def gray_thresh(img, thresh=(0, 255)): # thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    return binary_output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)): # thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, prev_source, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    ysize = img.shape[0]
    xsize = img.shape[1]

    low_angle = np.pi / 7.2  # 25 degrees
    high_angle = np.pi / 2.4  # 75 degrees
    low_threshold = np.tan(low_angle)
    high_threshold = np.tan(high_angle)
    l_x = []
    l_y = []
    r_x = []
    r_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if (slope) < 0:
                if (-1 * high_threshold) <= slope <= (-1 * low_threshold):
                    l_x.append(x1)
                    l_x.append(x2)
                    l_y.append(y1)
                    l_y.append(y2)
            elif (slope) > 0:
                if low_threshold <= slope <= high_threshold:
                    r_x.append(x1)
                    r_x.append(x2)
                    r_y.append(y1)
                    r_y.append(y2)

    l_x1 = 0
    l_y1 = 0
    l_x2 = 0
    l_y2 = 0
    r_x1 = 0
    r_y1 = 0
    r_x2 = 0
    r_y2 = 0

    if (len(l_x) > 0 and len(l_y) > 0 and len(r_x) > 0 and len(r_y) > 0):
        fit_left = np.polyfit(l_y, l_x, 1)
        fit_right = np.polyfit(r_y, r_x, 1)
        top_h_adjust = 90 # 100
        l_y1 = (ysize // 2) + top_h_adjust
        l_x1 = int(fit_left[0] * l_y1 + fit_left[1])
        l_y2 = ysize
        l_x2 = int(fit_left[0] * l_y2 + fit_left[1])

        r_y1 = (ysize // 2) + top_h_adjust
        r_x1 = int(fit_right[0] * r_y1 + fit_right[1])
        r_y2 = ysize
        r_x2 = int(fit_right[0] * r_y2 + fit_right[1])
        if (len(prev_source) != 0):
            top_src_offset = 25
            bot_src_offset = 100
            if (l_x1 < (prev_source[0][0] - top_src_offset) or l_x1 > (prev_source[0][0] + top_src_offset)):
                l_x1 = prev_source[0][0]
            if (l_y1 < (prev_source[0][1] - top_src_offset) or l_y1 > (prev_source[0][1] + top_src_offset)):
                l_y1 = prev_source[0][1]

            if (l_x2 < (prev_source[1][0] - bot_src_offset) or l_x2 > (prev_source[1][0] + bot_src_offset)):
                l_x2 = prev_source[1][0]
            if (l_y2 < (prev_source[1][1] - bot_src_offset) or l_y2 > (prev_source[1][1] + bot_src_offset)):
                l_y2 = prev_source[1][1]

            if (r_x1 < (prev_source[2][0] - top_src_offset) or r_x1 > (prev_source[2][0] + top_src_offset)):
                r_x1 = prev_source[2][0]
            if (r_y1 < (prev_source[2][1] - top_src_offset) or r_y1 > (prev_source[2][1] + top_src_offset)):
                r_y1 = prev_source[2][1]

            if (r_x2 < (prev_source[3][0] - bot_src_offset) or r_x2 > (prev_source[3][0] + bot_src_offset)):
                r_x2 = prev_source[3][0]
            if (r_y2 < (prev_source[3][1] - bot_src_offset) or r_y2 > (prev_source[3][1] + bot_src_offset)):
                r_y2 = prev_source[3][1]
    else:
        l_x1 = prev_source[0][0]
        l_y1 = prev_source[0][1]
        l_x2 = prev_source[1][0]
        l_y2 = prev_source[1][1]
        r_x1 = prev_source[2][0]
        r_y1 = prev_source[2][1]
        r_x2 = prev_source[3][0]
        r_y2 = prev_source[3][1]

    cv2.line(img, (l_x1, l_y1), (l_x2, l_y2), color, thickness)
    cv2.line(img, (r_x1, r_y1), (r_x2, r_y2), color, thickness)
#    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    src = np.float32([[l_x1, l_y1], [l_x2, l_y2], [r_x1, r_y1], [r_x2, r_y2]])

    left_adjust = 350
    right_adjust = 350
    top_adjust = 0
    dst = np.float32([[(xsize // 2) - left_adjust + top_adjust, 0],
                       [(xsize // 2) - left_adjust, ysize],
                       [(xsize // 2) + right_adjust - top_adjust, 0],
                       [(xsize // 2) + right_adjust, ysize]])
    return src, dst

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, prev_source):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    src, dst = draw_lines(line_img, lines, prev_source)
    return line_img, src, dst


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.

    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def warp(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv

def find_lines(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
#    plt.plot(histogram)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    y_eval = np.max(ploty)
    left_x_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    dist = distance_from_center(binary_warped.shape[1], left_x_pos, right_x_pos)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img, left_fit, right_fit, ploty, left_fitx, right_fitx, dist

def distance_from_center(width, l_x, r_x):
    xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
    dist = (r_x - width/2) - (width/2 - l_x)
    dist_m = dist * xm_per_pix
    return dist_m

def skip_sliding_window(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
    left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
    right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
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

    y_eval = np.max(ploty)
    left_x_pos = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_pos = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    dist = distance_from_center(binary_warped.shape[1], left_x_pos, right_x_pos)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fit, right_fit, ploty, left_fitx, right_fitx, dist

def measure_curvature(ploty, left_fitx, right_fitx):

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
#    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

def draw_lane_lines(undist, warped, Minv, left_fitx, right_fitx, ploty, curvature, distance):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    topCenterofText1 = (400, 100)
    topCenterofText2 = (400, 150)
    fontScale = 1
    fontColor = (255,255,255)
    lineType = 2
    curv_text = "Radius of Curvature = " + str(int(curvature)) + "(m)"
    dist_text = ""
    if (distance > 0):
        dist_text = "Vehicle is " + str(round(distance, 2)) + "m right of center"
    elif (distance < 0):
        dist_text = "Vehicle is " + str(round(abs(distance), 2)) + "m left of center"
    else:
        dist_text = "Vehicle is on the center"

    cv2.putText(result, curv_text, topCenterofText1, font, fontScale, fontColor, lineType)
    cv2.putText(result, dist_text, topCenterofText2, font, fontScale, fontColor, lineType)
#    plt.imshow(result)
#    plt.show()
    return result

def rgb_select(img, thresh=(0, 255)):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_output

def hls_select(img, h_thresh=(0, 255), s_thresh=(0, 255)):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_output = np.zeros_like(h_channel)
    h_output[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    s_output = np.zeros_like(s_channel)
    s_output[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    binary_output = h_output & s_output
#    binary_output = s_output
    return binary_output

def hsv_select(img, h_thresh=(0, 255), s_thresh=(0, 255)):
    img = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    h_output = np.zeros_like(h_channel)
    h_output[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    s_output = np.zeros_like(s_channel)
    s_output[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    binary_output = h_output & s_output
    return binary_output

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Read in the saved mtx & dist
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Read in an image
    img_size = (image.shape[1], image.shape[0])
    u_image = cal_undistort(image, mtx, dist)

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(u_image, orient='x', sobel_kernel=ksize, thresh=(30, 100)) # 20 30
    grady = abs_sobel_thresh(u_image, orient='y', sobel_kernel=ksize, thresh=(30, 100)) # 20 30
    mag_binary = mag_thresh(u_image, sobel_kernel=ksize, mag_thresh=(50, 100)) #30 50
    dir_binary = dir_thresh(u_image, sobel_kernel=ksize, thresh=(0.7, 1.3))

#    gray_binary = gray_thresh(u_image, thresh=(180, 255))
#    rgb_binary = rgb_select(u_image, thresh=(200, 255))
    hls_binary = hls_select(u_image, h_thresh=(15, 100), s_thresh=(170, 255)) # 90 170
#    hsv_binary = hsv_select(u_image, h_thresh=(15, 100), s_thresh=(170, 255))

    gradxy = (gradx == 1) & (grady == 1)
    mag_dir_binary = (mag_binary == 1) & (dir_binary == 1)
    g_m_binary = gradxy | mag_dir_binary

    c_binary = np.dstack((np.zeros_like(g_m_binary), g_m_binary, hls_binary)) * 255
    combined = np.zeros_like(dir_binary)
    combined[g_m_binary | (hls_binary == 1)] = 1

    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    #ax1.imshow(u_image)
    ax1.imshow(c_binary)
    #    ax1.imshow(hls_binary, cmap='gray')
    ax1.set_title('Stacked thresholds', fontsize=50)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Combined S channel and gradient thresholds', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    quit()
    '''
#    result_image = combined * 255
#    cv2.imwrite("combined_result.jpg", result_image)

    global isFirst
    global prev_src
    global vertices

    # Next we'll create a masked edges image using cv2.fillPoly()
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]

    top_w_adjust = 100  # at least 90
    top_h_adjust = 90  # 100
    top_y = (height // 2) + top_h_adjust
    if isFirst:
        bottom_h_adjust = 0 # 30
        bottom_w_adjust = 0

        left_top = ((width // 2) - top_w_adjust, top_y)
        right_top = ((width // 2) + top_w_adjust, top_y)
        left_bottom = (bottom_w_adjust, height - bottom_h_adjust)
        right_bottom = (width - bottom_w_adjust, height - bottom_h_adjust)
        vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
#        print("first vertices:"+str(vertices))
    else:
#        print("next vertices:" + str(vertices))
        pass

    masked_edges = region_of_interest(np.uint8(combined), vertices)
#    plt.imshow(masked_edges, cmap='gray')
#    plt.show()
#    quit()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # minimum number of pixels making up a line # 40          25
    max_line_gap = 20  # maximum gap in pixels between connectable line segments # 20

    # Run Hough on edge detected image
    line_image, src, dst = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, prev_src)
    prev_src = np.copy(src)
#    print("type_src:" +str(type(src)))
#    src = np.float32([[l_x1, l_y1], [l_x2, l_y2], [r_x1, r_y1], [r_x2, r_y2]])

    top_w_offset = 38 # 30 50
    top_h_offset = 0
    bottom_h_offset = 0
    bottom_w_offset = 100 # 90

    s_l_x1 = np.max(src[0][0] - top_w_offset, 0)
    s_l_y1 = src[0][1]
    s_l_x2 = np.max(src[1][0] - bottom_w_offset, 0)
    s_l_y2 = src[1][1]

    s_r_x1 = min(src[2][0] + top_w_offset, width)
    s_r_y1 = src[2][1]
    s_r_x2 = min(src[3][0] + bottom_w_offset, width)
    s_r_y2 = src[3][1]

    left_top = (s_l_x1, s_l_y1)
    right_top = (s_r_x1, s_r_y1)
#    left_top = ((width // 2) - top_w_adjust, top_y)
#    right_top = ((width // 2) + top_w_adjust, top_y)
    left_bottom = (s_l_x2, s_l_y2)
    right_bottom = (s_r_x2, s_r_y2)

#    x = [left_bottom[0], right_bottom[0], right_top[0], left_top[0], left_bottom[0]]
#    y = [left_bottom[1], right_bottom[1], right_top[1], left_top[1], left_bottom[1]]
#    plt.plot(x, y, 'b--', lw=4)

    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    warped, perspective_M, perspective_Minv = warp(combined, src, dst)

    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(combined, cmap='gray')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    '''
#    result_image = warped * 255
#    cv2.imwrite("warped-result.jpg", result_image)
    '''
    x = [left_bottom[0], right_bottom[0], right_top[0], left_top[0], left_bottom[0]]
    y = [left_bottom[1], right_bottom[1], right_top[1], left_top[1], left_bottom[1]]
    plt.plot(x, y, 'b--', lw=4)

    alpha = 0.8
    beta = 1.
    lambda_ = 0.
    lines_edges = weighted_img(line_image, u_image, alpha, beta, lambda_)
    plt.imshow(lines_edges)
    '''

    global l_fit
    global r_fit
    l_cur = 0
    r_cur = 0
    dist = 0
    if (isFirst):
        out_img_re, l_fit, r_fit, plot_y, l_fitx, r_fitx, dist = find_lines(warped)
        l_cur, r_cur = measure_curvature(plot_y, l_fitx, r_fitx)
        isFirst = False
    else:
        out_img_re, l_fit, r_fit, plot_y, l_fitx, r_fitx, dist = skip_sliding_window(warped, l_fit, r_fit)
        l_cur, r_cur = measure_curvature(plot_y, l_fitx, r_fitx)
    result = draw_lane_lines(u_image, warped, perspective_Minv, l_fitx, r_fitx, plot_y, l_cur, dist)
    return result

'''
in_img = mpimg.imread('project_v_images/frame0.jpg')
re_img = process_image(in_img)
cv2.imwrite("output_result.jpg", re_img)
plt.imshow(re_img)
plt.show()
quit()
'''

'''
#0 ~ 1251
for fno in range(895, 920):
    file_name = "frame"+str(fno)+".jpg"
    print(file_name)
    in_img = mpimg.imread('project_v_images/'+file_name)
    re_img = process_image(in_img)
    plt.imshow(re_img)
    plt.show()
quit()
'''

white_output = 'output_videos/project_video.mp4'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

clip1 = VideoFileClip("project_video.mp4")      # 1280 x 720
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)