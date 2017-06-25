import numpy as np
import cv2
from scipy.signal import butter, lfilter

# image display
import IPython.display
import PIL.Image
from io import BytesIO


# DATA PROCESSING TOOLS

def load_image(filename):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return image


def load_images(filenames):
    return [load_image(filename) for filename in filenames]


# IMAGE PROCESSING TOOLS


def pixel_normalization(p):
    return (p / 255.0) - 0.5


roi_points_forward = [[600, 310], [0, 620], [0, 680], [475, 680], [626, 491]]
roi_points_reversed = [(1280 - p[0], p[1]) for p in roi_points_forward[::-1]]
roi_points = np.array([roi_points_forward + roi_points_reversed], np.int32)


def butter_lowpass(cutoff, fs, order=5):
    '''
    low pass filter using scipy
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
    applies a low pass filter to the data, in a forward pass w/ specified params
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def region_of_interest(img, vertices=roi_points, invert_mask=False):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        zeros = (0,) * channel_count
    else:
        ignore_mask_color = 255
        zeros = 0

    # defining a blank mask to start with
    if invert_mask:
        mask = np.full(img.shape, np.int32(255))
    else:
        mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    if invert_mask:
        mask = cv2.fillPoly(mask, vertices, zeros)
    else:
        mask = cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img.astype(np.int32), mask.astype(np.int32))
    return masked_image.astype(np.uint8)


mtx = np.array([[1.15396100e+03, 0.00000000e+00, 6.69706490e+02],
                [0.00000000e+00, 1.14802504e+03, 3.85655584e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


dist = np.array([[-2.41018756e-01, -5.30666106e-02, -1.15811356e-03,
                  -1.28285248e-04, 2.67027151e-02]])


def restore_3channel(image):
    shape = image.shape
    if len(shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        return image


def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)


def gray(image):
    if len(image.shape) == 2:
        return image
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def bin_threshold(image, lower, upper):
    if len(image.shape) > 2:
        error = '{} is the wrong shape, should be single channel, single image.'.format(image.shape)
        raise NameError(error)

    binary_output = np.zeros_like(image)
    binary_output[(image >= lower) & (image <= upper)] = 255

    return binary_output


def scaled_sobel(image, directions=[(1, 0), (0, 1)]):
    if type(directions) != list:
        directions = [directions]

    sobels = [cv2.Sobel(gray(image), cv2.CV_64F, *direction) for direction in directions]
    scaled = [np.uint8(255 * np.abs(sobel) / np.max(sobel)) for sobel in sobels]

    return scaled


def sobel_x(image):
    x_sobel, = scaled_sobel(image, directions=(1, 0))
    return bin_threshold(x_sobel, 26, 150)


def sobel_y(image):
    y_sobel, = scaled_sobel(image, directions=(0, 1))
    return bin_threshold(y_sobel, 41, 201)


def mag_thresh(image, kernel_size=3, mag_thresh=(41, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Return the binary image
    return bin_threshold(gradmag, *mag_thresh)


def dir_threshold(image, kernel_size=3, thresh=(0.2 * np.pi, 0.4 * np.pi)):

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Return the binary image
    return bin_threshold(absgraddir, *thresh)


def find_lane_pixels(original_image):
    image = np.copy(original_image)
    gimage = gray(image)

    gradx = sobel_x(gimage)
    grady = sobel_y(gimage)
    magnitude = mag_thresh(gimage)
    direction = dir_threshold(gimage)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = (hls[:, :, 1] > 100) * 255 * 1
    s_channel = (hls[:, :, 2] > 100) * 255 * 1

    req_votes = 3
    votes = np.round((gradx + grady + magnitude +
                      direction + s_channel + l_channel) / 255.0)
    votes_thresholded = (votes >= req_votes).astype(np.uint8)
    roi = region_of_interest(votes_thresholded)

    return roi


source_points = np.array([[[684., 450.],
                           [594., 452.],
                           [255., 686.],
                           [1053., 686.]]], dtype=np.float32)


def perspective_transformation(source_points):
    px_per_meter = 720 / (3 * 14.64)
    x_stretch_factor = 5
    width_in_px = 3.7 * px_per_meter * x_stretch_factor

    left_x = 640 - width_in_px / 2
    right_x = left_x + width_in_px
    top_y = 0
    bottom_y = 720

    destination_points = np.array([[[right_x, top_y], [left_x, top_y],
                                    [left_x, bottom_y], [right_x, bottom_y]]]).astype(np.float32)

    M = cv2.getPerspectiveTransform(source_points, destination_points)
    Minv = cv2.getPerspectiveTransform(destination_points, source_points)

    return M, Minv


pM = np.array([[-2.69603747e-01, -1.46703503e+00, 7.88103005e+02],
               [-4.02906609e-02, -1.87351573e+00, 8.70721473e+02],
               [-5.59592513e-05, -2.29695157e-03, 1.00000000e+00]])


pinvM = np.array([[2.83609373e-01, -7.69506913e-01, 4.46512989e+02],
                  [-1.89109756e-02, -5.05610926e-01, 4.55150259e+02],
                  [-2.75670229e-05, -1.20442502e-03, 1.00000000e+00]])


def undistort_find_px_transform(image):
    lane_pixels = find_lane_pixels(image)
    undist = undistort(lane_pixels * 255)
    unwarped = cv2.warpPerspective(undist, pM, (image.shape[1], image.shape[0]))

    return unwarped


def compute_lane_lines(lanepx):
    # TODO - break this beast up!
    binary_warped = (lanepx > 150)
    height, width = binary_warped.shape

    histogram = np.sum(binary_warped[height // 2:, :], axis=0)
    fwd_pass_smoothing = butter_lowpass_filter(histogram, 3, 100, order=2)
    smoothed_histogram = butter_lowpass_filter(fwd_pass_smoothing[::-1], 2, 100, order=2)[::-1]

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img.astype('uint8')

    midpoint = np.int(smoothed_histogram.shape[0] / 2)
    leftx_base = np.argmax(smoothed_histogram[:midpoint])
    rightx_base = np.argmax(smoothed_histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(height / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    draw_img = np.copy(out_img).astype('uint8')

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(draw_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(draw_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
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
    ploty = np.linspace(0, binary_warped.shape[0] - 1, 50)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    left_line = np.dstack((left_fitx, ploty)).astype('int32')

    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    right_line = np.dstack((right_fitx, ploty)).astype('int32')

    draw_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    draw_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    cv2.polylines(draw_img, [left_line, right_line], 0, color=(255, 255, 0), thickness=3)

    return draw_img, left_fit, right_fit, left_line, right_line, left_fitx, right_fitx, ploty


# CURVATURE

def curvature(A, B, y):
    numerator = (1 + (2 * A * y + B) ** 2) ** (1.5)
    denominator = 2 * A
    return numerator / denominator


def curvature_via_fit(fit, y):
    return curvature(fit[0], fit[1], y)


# DATA INVESTIGATION TOOLS

def showarray(a, fmt='png', width=None, height=None):
    '''
    Displays an image without the ugliness of matplotlib
    '''
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue(), width=width, height=height))


def side_by_side(image, transformation, vert=False, twidth=None, theight=None):
    image_copy = np.copy(image)
    image_copy = restore_3channel(image_copy)
    transformed_image = restore_3channel(transformation(image_copy))

    if vert:
        display_image = np.vstack((image_copy, transformed_image))
    else:
        display_image = np.hstack((image_copy, transformed_image))

    showarray(display_image, width=twidth, height=theight)


def side_by_sides(images, transformation, vert=False, twidth=None, theight=None):
    for image in images:
        side_by_side(image, transformation, vert=vert, twidth=twidth, theight=theight)
