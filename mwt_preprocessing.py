##
##  Near-shore Wave Tracking
##  mwt_preprocessing.py
##
##  Created by Justin Fung on 9/1/17.
##  Copyright 2017 justin fung. All rights reserved.
##
## ========================================================

"""Routine for preprocessing video frames.

 Method of preprocessing is:
 -1. resize image
 -2. extract foreground
 -3. denoise image
"""

from __future__ import division

import cv2
import math
import numpy as np
import copy

# Resize factor (downsize) for analysis:
RESIZE_FACTOR = 0.25

# Number of frames that constitute the background history:
BACKGROUND_HISTORY = 900

# Number of gaussians in BG mixture model:
NUM_GAUSSIANS = 5

# Minimum percent of frame considered background:
BACKGROUND_RATIO = 0.7

# Morphological kernel size (square):
MORPH_KERN_SIZE = 3

# Init the background modeling and foreground extraction mask.
mask = cv2.bgsegm.createBackgroundSubtractorMOG(
                                  history=BACKGROUND_HISTORY,
                                  nmixtures=NUM_GAUSSIANS,
                                  backgroundRatio=BACKGROUND_RATIO,
                                  noiseSigma=0)

# Init the morphological transformations for denoising kernel.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   (MORPH_KERN_SIZE, MORPH_KERN_SIZE))


def euc(p1, p2):
    return math.sqrt((p1[1]-p2[1])**2 + (p1[0]-p2[0])**2)

def scale_up_rect(rect):
    # find the slope between bottom left and top right
    m_diag_1 = (rect[2][1] - rect[0][1]) / (rect[2][0] - rect[0][0])
    # find slope between top left and bottom right
    m_diag_2 = (rect[1][1] - rect[3][1]) / (rect[1][0] - rect[3][0])

    delta_x = (rect[3][0] - rect[1][0]) * 0.25
    delta_y = (rect[0][1] - rect[1][1]) * 0.75

    new_rect = copy.deepcopy(rect)
    # use some math to figure out how far down the line we need to 'walk'
    new_rect[0][0] -= delta_x/math.sqrt(1 + m_diag_1**2)
    new_rect[0][1] += m_diag_1 * (new_rect[0][0] - rect[0][0]) + delta_y

    new_rect[1][0] -= delta_x/math.sqrt(1 + m_diag_2**2)
    new_rect[1][1] += m_diag_2 * (new_rect[1][0] - rect[1][0]) - delta_y

    new_rect[2][0] += delta_x/math.sqrt(1 + m_diag_1**2)
    new_rect[2][1] += m_diag_1 * (new_rect[2][0] - rect[2][0]) - delta_y

    new_rect[3][0] += delta_x/math.sqrt(1 + m_diag_2**2)
    new_rect[3][1] += m_diag_2 * (new_rect[3][0] - rect[3][0]) + delta_y
    
    return new_rect


def rotate(box, img):
    """box is the four corner of the rectangle we wish to rotate in the order
    of bottom left top left top right bottom right."""
    """All we need to do now is to crop the black part (end of frame) 
    from the image. Rotate is working as is the display portion"""
    
    
    #edit this to correctly reflect width and height"
    box[:] = [(1/RESIZE_FACTOR)*box[i] for i in range(4)]

    box = scale_up_rect(box)

    width = int(euc(box[1], box[2]))
    height = int(euc(box[0], box[1]))

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def _resize(frame):
    """Resizing function utilizing OpenCV.

    Args:
      frame: A frame from a cv2.video_reader object to process

    Returns:
      resized_frame: the frame, resized
    """
    resized_frame = cv2.resize(frame,
                               None,
                               fx=RESIZE_FACTOR,
                               fy=RESIZE_FACTOR,
                               interpolation=cv2.INTER_AREA)

    return resized_frame


def preprocess(frame):
    """Preprocesses video frames through resizing, background
    modeling, and denoising.

    Args:
      input: A frame from a cv2.video_reader object to process

    Returns:
      output: the preprocessed frame
    """

    # 1. Resize the input.
    output = _resize(frame)

    # 2. Model the background and extract the foreground with a mask.
    output = mask.apply(output)

    # 3. Apply the morphological operators to suppress noise.
    output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)

    return output
