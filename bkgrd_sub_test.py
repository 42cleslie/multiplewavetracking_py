import sys
import getopt
import time
import os
import numpy as np
import random as rng
import cv2
import copy

import mwt_detection

OUTPUT_DIR = "output"

STACKED_WAVE_FILE = "background_stack2"

RESIZE_FACTOR = 0.25

def status_update(frame_number, tot_frames):
    """A simple inline status update for stdout.
    Prints frame number for every 100 frames completed.

    Args:
      frame_number: number of frames completed
      tot_frames: total number of frames to analyze

    Returns:
      VOID: writes status to stdout
    """
    if frame_number == 1:
        print("Starting analysis of %d frames..." %tot_frames)

    if frame_number % 100 == 0:
        print("%d" %frame_number, end='')
        sys.stdout.flush()
    elif frame_number % 10 == 0:
        print(".", end='')
        sys.stdout.flush()

    if frame_number == tot_frames:
        print ("End of video reached successfully.")

def explode(flat_frame, ref_frame):
    exploded_frame = np.zeros_like(ref_frame)
    exploded_frame[:,:,0] = flat_frame
    exploded_frame[:,:,1] = flat_frame
    exploded_frame[:,:,2] = flat_frame

    return exploded_frame

def add_contours(frame):
    contours, _ = cv2.findContours(image=frame, mode=cv2.RETR_EXTERNAL,
                    method=cv2.CHAIN_APPROX_NONE, hierarchy=None, offset=None)

    drawing = explode(copy.deepcopy(frame), np.zeros((frame.shape[0], frame.shape[1], 3)))

    for cnt in contours:
        rect = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))
        
        drawing_color = (0, rng.randint(0,256), 0) # green
        if mwt_detection.keep_contour(cnt, area=True, inertia=True):
            val = rng.randint(0, 256)
            drawing_color = (val,0,val) # purple

        elif mwt_detection.keep_contour(cnt, area=False, inertia=True):
            drawing_color = (rng.randint(0,256), 0, 0) #blue

        elif mwt_detection.keep_contour(cnt, area=True, inertia=False):
            drawing_color = (0, 0, rng.randint(0,256)) # red

        cv2.drawContours(drawing, [rect], 0, drawing_color, 2)
    
    return drawing.astype("uint8")

def main(argv):
    """main"""
    # The command line should have one argument-
    # the name of the videofile.
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv, "i:h:m:r:k:")
    except getopt.GetoptError:
        print ("usage: mwt.py -i <inputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == ("-i"):
            inputfile = arg
        elif opt == ("-h"):
            hist = int(arg)
        elif opt == ("-m"):
            mix = int(arg)
        elif opt == ("-r"):
            rat = float(arg)
        elif opt == ("-k"):
            kern = int(arg)

    # Read video.
    print ("Checking video from", inputfile)
    inputvideo = cv2.VideoCapture(inputfile)

    # Exit if video cannot be opened.
    if not inputvideo.isOpened():
        sys.exit("Could not open video.")

    mask = cv2.bgsegm.createBackgroundSubtractorMOG(
                                    history=hist,
                                    nmixtures=mix,
                                    backgroundRatio=rat,
                                    noiseSigma=0)

    original_width = inputvideo.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_FACTOR
    original_height = inputvideo.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_FACTOR
    fps = inputvideo.get(cv2.CAP_PROP_FPS)

    # Make an output directory if necessary.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Initiate video writer object by defining the codec and initiating
    # the VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outFile = STACKED_WAVE_FILE + "_h" + str(hist) + "_m" + str(mix) + "_r" + str(rat) + "_k" + str(kern) + "cnt.mp4"
    output_path = os.path.join(OUTPUT_DIR, outFile)
    print("Writing video to", output_path)
    out = cv2.VideoWriter(output_path,
                          fourcc,
                          fps,
                          (int(original_width * 2), int(original_height * 3)),
                          True)

    frame_num = 1
    num_frames = int(inputvideo.get(cv2.CAP_PROP_FRAME_COUNT))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   (kern, kern))
    while True:

        status_update(frame_num, num_frames)

        successful_read, original_frame = inputvideo.read()
        if not successful_read:
            break

        resized_frame = cv2.resize(original_frame, None, fx=RESIZE_FACTOR,
                        fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

        masked_frame = mask.apply(resized_frame)
        masked_image = explode(masked_frame, resized_frame)

        morph_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_OPEN, kernel)
        morph_image= explode(morph_frame, resized_frame)

        orig_cnt = np.zeros_like(resized_frame)#add_contours(resized_frame)
        masked_cnt = add_contours(masked_frame)
        morph_cnt = add_contours(morph_frame)

        comb_left = np.concatenate((masked_image, resized_frame, morph_image), axis=0)
        comb_right = np.concatenate((masked_cnt, orig_cnt, morph_cnt), axis=0)
        comb = np.concatenate((comb_left, comb_right), axis=1)

        out.write(comb)
        frame_num += 1
        
    # Clean-up resources.
    inputvideo.release()
    out.release()

if __name__ == "__main__":
    main(sys.argv[1:])