#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#
##########################################################################################
#
from __future__ import print_function
from __future__ import division
from collections import deque

import sys
import random
import os
import argparse

import datetime
import time

#import re
#import io
#import glob

import math
import numpy as np

###########################
# conditional imports ...
try:
    import pygame
    from pygame.locals import *
#    from pygame import camera
except ImportError:
    pass
    print("pyGame Import Error!!!")
    print("     EXIT program here ...")
    sys.exit(1)
#
try:
    import serial;
    import serial.tools.list_ports;
except ImportError:
    pass
    print("serial Import Error!!!")
    print("     EXIT program here ...")
    sys.exit(1)
#    propably using a different library for the camera
#try:
#    import cv2
#except importerror:
#    pass
#    print("opencv import error!!!")
#    print("     exit program here ...")
#    sys.exit(1)
#
#try:
#    import imutils
#    from imutils.object_detection import non_max_suppression
#    from imutils import paths
#except ImportError:
#    pass
#    print("imutils Import Error!!!")
#    print("     EXIT program here ...")
#    sys.exit(1)
#
#
##########################################################################################



#
#
##########################################################################################
#   global variables / constants
#
#######################################
#   create list/dictionary with face expressions and elements ...
#   unfortunately python dictionarys get dis-ordered during processing -
#   to be able to keep the sequence of layers right, they have to be sortable!
#
homeDir = os.getcwd()   # check for home directory
fileExt = '.png'        # select file type
exprsList = {}          # Create an empty dict
#
#   create a nested dictionary representing the folder structure of homeDir
#def get_directory_structure(homeDir):
#    homeDir = homeDir.rstrip(os.sep)
#    start = homeDir.rfind(os.sep) + 1
#    for path, dirs, files in os.walk(homeDir):
#        folders = path[start:].split(os.sep)
#        subdir = dict.fromkeys(files)
#        parent = reduce(dict.get, folders[:-1], exprsList)
#        parent[folders[-1]] = subdir
#    return exprsList
#
for path, dirs, files in os.walk(homeDir) :
    #   select subdirectorys by relevance
    for dir in dirs :
        if 'gfx' in dir :
            exprsList[ dir[4:] ] = {}
            #   select files via extension
            for file in os.listdir( os.path.join(homeDir,dir) ) :
                if fileExt in file :
                    exprsList[ dir[4:] ][file] = os.path.join('./',dir,file)
# print(exprsList)
#
#######################################
#
eyesMatch = '03_Auge'    # sub string pattern for matching eye elements from dictinary ...
#   image library - loading images only once ...
imageList = {}
#   scale factor for screen size ...
screen_scalefactor = 1
screen_w = int(  720*screen_scalefactor )
screen_h = int( 1280*screen_scalefactor )
#   time-ranges for eye-blinks ...
eyeRangeBlnkTme = ( 4*1000, 8*1000 )   #  Intervall [4.0s-8.0s]
eyeRangeBlnkDur = ( 1*100 , 2*100  )   #  Durations [0.1s-0.2s]
#   scaling the maximum range of eye movements as elliptic paths ...
eyeRadiusEllpsX = int(  46*screen_scalefactor )
eyeRadiusEllpsY = int(  20*screen_scalefactor )
#   -> milliseconds clock.tick(fps) once per frame, the program will never run at more than 'fps' frames per second ...
fps = 30
pointXY = (0,0)
facePointXY = (0,0)
headRange = {"x": {"min": -60.0, "max": 60.0}, "y": {"min": 0.0, "max": 180.0}}
serialSendInterval = 100 # -> minimum 200ms between the values send via serial port
serialDeviceName = "Arduino Micro"
serialBaudrate = 9600
current_milli_time = lambda: int(round(time.time() * 1000))
trimTupleTo = lambda values, low, high: tuple(value if value > low else low for value in (value if value < high else high for value in values))
differenceThreshold = 14

#
#######################################
#
#if cv2 :
#    # set HAARCASCADES
#    face_cascade        = cv2.CascadeClassifier( './resources/haarcascades/haarcascade_frontalface_default.xml' )
#    eye_cascade         = cv2.CascadeClassifier( './resources/haarcascades/haarcascade_eye.xml'                 )
#    apple_cascade       = cv2.CascadeClassifier( './resources/apfel_classifier.xml'                             )
#    banana_cascade      = cv2.CascadeClassifier( './resources/banana_classifier.xml'                            )
#    #
#    # set HOG descriptor/person detector
#    hog                 = cv2.HOGDescriptor()
#    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
#    #
#    # set MOG background subtractor ...
#    fgbg                = cv2.bgsegm.createBackgroundSubtractorMOG()
#
#
##########################################################################################



#
#
##########################################################################################
#   creating a class for opencv tracking business ...
class Track(object):

    #   constructor ...
    def __init__(self, camID):
        self.camID = camID
        self.ratio_h = 1.0
        self.ratio_w = 1.0

        if 'kinect' == self.camID :
            self.camera = cv2.VideoCapture(self.camID)

        elif 'stereo' == self.camID :
            self.cameraL = cv2.VideoCapture(0)
            self.cameraR = cv2.VideoCapture(1)

        else :
            self.camera = cv2.VideoCapture(self.camID)
#           self.camera.set(3, screen_w)
#           self.camera.set(4, screen_h)
#           self.cam_w = self.camera.get(3)
#           self.cam_h = self.camera.get(4)



#######################################
    #   function to convert from opencv-frame to pygame surface image for blitting ...
    def convertFrame2Surface(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        surface = pygame.surfarray.make_surface( frame )
        return surface
    
    
#######################################
#   webcam feed, return image ...
    def getRGBFrame(self):
        # grab the current frame
        (grabbed, frame) = self.camera.read()
        # resize the frame ...
        frame = imutils.resize(frame,  width=screen_w)
        # get the ratio ...
        frame_h = np.size(frame, 0)
        frame_w = np.size(frame, 1)
        self.ratio_h = (frame_h/screen_h)
        self.ratio_w = (frame_w/screen_w)
        # flip the frame ... for getting the coordinates straight ...
        frame = cv2.flip(frame,1)
        return frame


#######################################
#   double webcam feed, return imageL and imageR ...
    def getStereoRGBFrame(self):
        # put here the rgb frames together ...
        (grabbedL, frameL) = self.cameraL.read()
        (grabbedR, frameR) = self.cameraR.read()
        # resize the frame ...
        frameL = imutils.resize(frameL,  width=screen_w)
        frameR = imutils.resize(frameR,  width=screen_w)
        # get the ratio ...
        frame_h = np.size(frameL, 0)
        frame_w = np.size(frameL, 1)
        self.ratio_h = (frame_h/screen_h)
        self.ratio_w = (frame_w/screen_w)
        # flip the frame ... for getting the coordinates straight ...
        frameL = cv2.flip(frameL,1)
        frameR = cv2.flip(frameR,1)
        return frameL, frameR




#'''
#Simple example of stereo image matching and point cloud generation.
#Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
#'''
## Python 2/3 compatibility
#from __future__ import print_function
#
#import numpy as np
#import cv2
#
#ply_header = '''ply
#format ascii 1.0
#element vertex %(vert_num)d
#property float x
#property float y
#property float z
#property uchar red
#property uchar green
#property uchar blue
#end_header
#'''
#
#def write_ply(fn, verts, colors):
#    verts = verts.reshape(-1, 3)
#    colors = colors.reshape(-1, 3)
#    verts = np.hstack([verts, colors])
#    with open(fn, 'wb') as f:
#        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
#        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
#
#
#if __name__ == '__main__':
#    print('loading images...')
#    imgL = cv2.pyrDown( cv2.imread('../data/aloeL.jpg') )  # downscale images for faster processing
#    imgR = cv2.pyrDown( cv2.imread('../data/aloeR.jpg') )
#
#    # disparity range is tuned for 'aloe' image pair
#    window_size = 3
#    min_disp = 16
#    num_disp = 112-min_disp
#    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
#        numDisparities = num_disp,
#        blockSize = 16,
#        P1 = 8*3*window_size**2,
#        P2 = 32*3*window_size**2,
#        disp12MaxDiff = 1,
#        uniquenessRatio = 10,
#        speckleWindowSize = 100,
#        speckleRange = 32
#    )
#
#    print('computing disparity...')
#    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#
#    print('generating 3d point cloud...',)
#    h, w = imgL.shape[:2]
#    f = 0.8*w                          # guess for focal length
#    Q = np.float32([[1, 0, 0, -0.5*w],
#                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#                    [0, 0, 0,     -f], # so that y-axis looks up
#                    [0, 0, 1,      0]])
#    points = cv2.reprojectImageTo3D(disp, Q)
#    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
#    mask = disp > disp.min()
#    out_points = points[mask]
#    out_colors = colors[mask]
#    out_fn = 'out.ply'
#    write_ply('out.ply', out_points, out_colors)
#    print('%s saved' % 'out.ply')
#
#    cv2.imshow('left', imgL)
#    cv2.imshow('disparity', (disp-min_disp)/num_disp)
#    cv2.waitKey()
#    cv2.destroyAllWindows()



#######################################
# disparity = x − x′= Bf/Z
# x and x′ are the distance between points in image plane corresponding to the scene point 3D and their camera center
# B is the distance between two cameras (which we know) and f is the focal length of camera (already known)
# short: equation says that the depth of a point in a scene is inversely proportional to the difference in distance of corresponding image points and their camera centers.
# with this information, we can derive the depth of all pixels in an image ...
#
# OpenCV samples contain an example of generating disparity map and its 3D reconstruction. Check stereo_match.py in OpenCV-Python samples
#
#   double webcam feed, return image ...
    def getStereoDepthFrame(self):
        # put here the depth frame together ...
        frameL, frameR = getStereoRGBFrame()
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(frameL,frameR)

        try:
            #                from matplotlib import pyplot as plt
            self.matplotlib             = __import__('matplotlib')
            #                self.pyplot                 = self.matplotlib.pyplot
            self.plt                    = self.matplotlib.pyplot
        except ImportError:
            pass
            print("Import Error - no matplotlib!!!")
            sys.exit(1)

        plt.imshow(disparity,'gray')
        plt.show()

        return disparity


#######################################
# filter for max RGB values and return color enhanced image ...
#    For each pixel in the image I:
#    Grab the r, g, and b pixel intensities located at I[x, y]
#    Determine the maximum value of r, g, and b: m = max(r, g, b)
#    If r < m: r = 0
#    If g < m: g = 0
#    If b < m: b = 0
#    Store the r, g, and b values back in image: I[x, y] = (r, g, b)
    def maxRGBfilter(self, image):
        return self.maxRGBfilterCl( self.getRGBFrame(), 'None' )

#################
# filter for max RGB values and return color enhanced image ...
    def maxRGBfilterCl(self, image, colour):
        # split the image into its BGR components
        (B, G, R) = cv2.split(image)
        if colour is not None and 'blue' in colour:
            M = np.maximum(R, R)
            R[R < M] = 0
            G[G < M] = 255
            B[B < M] = 0
        elif colour is not None and 'green' in colour:
            M = np.maximum(R, R)
            R[R < M] = 0
            G[G < M] = 0
            B[B < M] = 255
        elif colour is not None and 'red' in colour:
            M = np.minimum(B, G)
            R[R < M] = 0
            G[G < M] = 255
            B[B < M] = 255
        elif colour is not None and 'yellow' in colour:
            M = np.maximum(B, G)
            R[R < M] = 255
            G[G < M] = 0
            B[B < M] = 0
        else :
        # find the maximum pixel intensity values for each (x, y)-coordinate, then set all pixel values less than M to zero
            M = np.maximum(np.maximum(R, G), B)
            R[R < M] = 0
            G[G < M] = 0
            B[B < M] = 0

        # put channels back together and return as image
        return cv2.merge([B, G, R])


#######################################
#   detect edges and return image with edges ...
    def maskAutoCanny(self, image, sigma=0.33):
        # convert image to grayscale, and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(  0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        return edged


#######################################
#   detect foreground and return masked image...
    def maskBackground(self):
        frame = self.getRGBFrame()
        fgmask = fgbg.apply(frame)
        masked = cv2.bitwise_and(frame, frame, mask = fgmask)
        return masked


#######################################
#   detect skin and return masked image ...
    def maskSkin(self):
        frame = self.getRGBFrame()

        # define the upper and lower boundaries of the HSV pixel intensities to be considered 'skin'
        lower = np.array([  0,  48,  80], dtype = "uint8")
        upper = np.array([ 20, 255, 255], dtype = "uint8")

        # resize the frame, convert it to the HSV color space, and determine the HSV pixel intensities that fall into the speicifed upper and lower boundaries
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # blur the mask, removing a little bit of noise, then apply mask ...
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        return skin


#######################################
#   detect contours and return contours ...
    def getContours(self):
        frame = self.getRGBFrame()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        skin_ycrcb_mint = np.array((0, 133, 77))
        skin_ycrcb_maxt = np.array((255, 173, 127))
        skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

        #CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
        image, contours, hierarchy = cv2.findContours(skin_ycrcb, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
        
        #CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
        #        image, contours, hierarchy = cv2.findContours(skin_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        maxAreaIdx = 0
        maxArea    = 0
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 500:
                cv2.drawContours(frame, contours, i, (255, 0, 0), 1)
            # find contour with biggest area ...
            if area > maxArea:
                maxAreaIdx  = i
                maxArea     = area
        # draw maxArea contours ...
        cv2.drawContours(frame, contours, maxAreaIdx, (255, 0, 0), 3)

        # show the frame to screen ...
        if args.camera_switch : cv2.imshow("detectContours", frame)

        return contours


#######################################
#   detect CAMshift (Continuously Adaptive Meanshift) motions and return...
    def detectCAMshift(self):
        (x, y) = (ret_x, ret_y) = pointXY

        # take first frame of the video
        frame = self.getRGBFrame()

        # setup initial location of window
        r,h,c,w = 100,100,100,100  # simply hardcoded values

        track_window = (c,r,w,h)
        # set up the ROI for tracking
        roi = frame[r:r+h, c:c+w]
#        roi = self.maskSkin()

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

#        while (1) :
        if(1) :
            frame = self.getRGBFrame()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)

            cv2.imshow('CAMshift',img2)

#            key = cv2.waitKey(delay=1)
#            if key == ord('q'):
#                break

        return (ret_x, ret_y)


#######################################
#   detect Dense Optical Flow motions and return...
    def detectDenseOptiFlow(self):
        (x, y) = (ret_x, ret_y) = pointXY

        frame1 = self.getRGBFrame()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

#        while (1) :
        if(1) :
            frame2 = self.getRGBFrame()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('DenseOpticalFlow',bgr)

            # Now update the previous frame
            prvs = next

#            key = cv2.waitKey(delay=1)
#            if key == ord('q'):
#                break

        return (ret_x, ret_y)


#######################################
#   detect Lucas-Kanade Optical Flow motions and return...
    def detectLKOptiFlow(self):
        (x, y) = (ret_x, ret_y) = pointXY

        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.3,
                                minDistance = 7,
                                blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        old_frame = self.getRGBFrame()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

#        while(1):
        if (1) :
            frame = self.getRGBFrame()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            if p1 is not None :
                good_new = p1[st==1]
                good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)

            # show the frame to screen ...
            if args.camera_switch :
                img = cv2.add(frame,mask)
                cv2.imshow('DenseOpticalFlow',img)

#            key = cv2.waitKey(delay=1)
#            if key == ord('q'):
#                break

        return (ret_x, ret_y)


#######################################
#   detect motions and return coordinates...
    def detectMotions(self, motionOption):
        (x, y) = (ret_x, ret_y) = pointXY

        avgFrame = self.getRGBFrame()
        avgFrame = cv2.cvtColor(avgFrame, cv2.COLOR_BGR2GRAY)
        avgFrame = cv2.GaussianBlur(avgFrame, (21, 21), 0)

#        avgFrame = None
#        count = 0
#        while (count < 5):
#            count = count + 1
#            avgFrame = avgFrame.copy().astype("float")
#            self.camera.truncate(0)

        frame = self.getRGBFrame()
        # frame, convert it to grayscale, and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
        # compute the absolute difference between the current frame and reference frame
        # delta = |background_model – current_frame|
        frameDelta = cv2.absdiff(avgFrame, gray)

        # accumulate the weighted average between the current frame and previous frames, then compute the difference between the current frame and running average
#        cv2.accumulateWeighted(gray, avgFrame, 0.5)
#        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avgFrame))

        # threshold the delta image, dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        image, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#        minArea = 500
#        # loop over the contours
#        for c in cnts:
#            # if the contour is too small, ignore it
#            if cv2.contourArea(c) < minArea:
#                continue
#            # draw the contour itself
#            cv2.drawContours(frame, [c], 0, (255, 255, 0), 1)
#            # compute the bounding box for the contour, draw it on the frame, and update the text
#            (x, y, w, h) = cv2.boundingRect(c)
#            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(frame, [c], 0, (255, 0, 0), 1)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # visualise the tracked object ...
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 1)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                ret_x = (x/self.ratio_w)
                ret_y = (y/self.ratio_h)

        # show the frame to screen ...
        if args.camera_switch :
#        # draw timestamp on the frame ...
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tmpImg = np.zeros_like(frame)
            tmpImg[:,:,0] = frameDelta
            tmpImg[:,:,1] = frameDelta
            tmpImg[:,:,2] = frameDelta
            all_img = np.concatenate((tmpImg, frame), axis=1)
            cv2.imshow('MotionDetect', all_img)

            moMask = cv2.bitwise_and(frame, frame, mask = frameDelta)
            cv2.imshow('MotionDetect2', moMask)
        
        # return values as point ...
        return ( ret_x, ret_y )


#######################################
#   detect skin, then hand and return coordinates...
    def detectHands(self, handOption):
        (x, y) = (ret_x, ret_y) = pointXY

        frame = self.getRGBFrame()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        image, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(frame.shape,np.uint8)

        max_area=0
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
        cnt=contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00

        centr=(cx,cy)
        cv2.circle(frame,centr,5,[0,0,255],2)
        cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
        cv2.drawContours(drawing,[hull],0,(0,0,255),2)

        cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt,returnPoints = False)

#        while(1):
        if(1):
            defects = cv2.convexityDefects(cnt,hull)
            mind=0
            maxd=0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                dist = cv2.pointPolygonTest(cnt,centr,True)
                cv2.line(frame,start,end,[0,255,0],2)
                cv2.circle(frame,far,5,[0,0,255],-1)

#            print("number defects: ",i)

#            key = cv2.waitKey(delay=1)
#            if key == ord('q'):
#                break

        # show the frame to screen ...
        if args.camera_switch :
            all_img = np.hstack((drawing, frame))
            if args.camera_switch : cv2.imshow('detectHands', all_img)


#######################################
#   detect gestures ...
    def detectGestures(self, gestureOption):
        (x, y) = (ret_x, ret_y) = pointXY

        if gestureOption is not None and 'hand' in gestureOption :
            track.detectHands()
            return pointXY

        elif gestureOption is not None and 'gesture' in gestureOption :
            track.detectMotions('motionOption')
            return pointXY

        frame = self.getRGBFrame()

#        crop_img = self.maskSkin()
        crop_img = self.getRGBFrame()

        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        (version, _, _) = cv2.__version__.split('.')
        if version is '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version is '2':
                   contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
        cv2.drawContours(drawing,[hull],0,(0,0,255),0)
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img,far,1,[0,0,255],-1)
            #dist = cv2.pointPolygonTest(cnt,far,True)
            cv2.line(crop_img,start,end,[0,255,0],2)
            #cv2.circle(crop_img,far,5,[0,0,255],-1)

#        if count_defects == 1:
#            cv2.putText(frame,"'ello !!!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
#        elif count_defects == 2:
#            str = "basic hand gesture recognizer ..."
#            cv2.putText(frame, str, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#        elif count_defects == 3:
#            cv2.putText(frame,"ecco !!!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
#        elif count_defects == 4:
#            cv2.putText(frame,"alora !!!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
#        else:
#            cv2.putText(frame,"???", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        # show the frame to screen ...
        if args.camera_switch :
            all_img = np.hstack((drawing, crop_img))
            all_img = np.hstack((all_img, frame))
            cv2.imshow('webCam', all_img)

        return ( ret_x, ret_y )


#######################################
    #   track a ball in front of a webcam and return coordinates ...
    def detectBall(self, colour):
        (x, y) = (ret_x, ret_y) = pointXY

        frame = self.getRGBFrame()
#        frame = self.maxRGBfilterCl( self.getRGBFrame(), colour )
#        frame = self.maxRGBfilter( self.getRGBFrame() )
        # define the lower and upper boundaries of the "colour" of the ball in the HSV color space ...
        if colour is not None and 'blue' in colour :
            #blue ...
            hsvLower = (100, 185,  35)
            hsvUpper = (120, 255, 110)
        elif colour is not None and 'green' in colour:
            #green ...
            hsvLower = ( 80, 125,  30)
            hsvUpper = (105, 255, 255)
        elif colour is not None and 'red' in colour:
            #red ...
            hsvLower = (120, 120,  50)
            hsvUpper = (255, 255, 255)
        elif colour is not None and 'yellow' in colour:
            #yellow ...
            hsvLower = (  0,  55,   0)
            hsvUpper = ( 64, 255, 255)
        else :
#            blue  = np.uint8([[[ 255,  0,  0 ]]])
#            green = np.uint8([[[   0,255,  0 ]]])
#            red   = np.uint8([[[   0,  0,255 ]]])
#            hsv_blue  = cv2.cvtColor(blue,  cv2.COLOR_BGR2HSV)
#            hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
#            hsv_red   = cv2.cvtColor(red,   cv2.COLOR_BGR2HSV)
#            print(hsv_blue, hsv_green, hsv_red)
#            [H-10, 100,100] and [H+10, 255, 255]
            hsvLower = (  0, 100, 100)
            hsvUpper = (255, 255, 255)
        # blur it ...
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # convert it to the HSV colour space ...
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # construct a mask for the colour, a series of dilations and erosions to remove any small blobs left in the mask
        mask = cv2.inRange(hsv, hsvLower, hsvUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # visualise the tracked object ...
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                ret_x = (x/self.ratio_w)
                ret_y = (y/self.ratio_h)
        # show the frame to screen ...
        if args.camera_switch : cv2.imshow("cam", frame)
        # return values as point ...
        return ( ret_x, ret_y )


#######################################
#   track an apple in front of cam and return coordinates ...
    def detectApple(self, appleOption):
        frame = self.getRGBFrame()
        (x, y) = (ret_x, ret_y) = pointXY

        #   here start tracking ...
        blurred = cv2.GaussianBlur(frame, (11,11), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#        apples = apple_cascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30) )
        apples = apple_cascade.detectMultiScale( gray,1.2, 4, cv2.CASCADE_SCALE_IMAGE, (35,25))

        # mark apples with rectangle around ...
        for (x, y, w, h) in apples:
            cv2.circle(frame, (int(x+w/2), int(y+h/2)), int(w/2), (0, 255, 0), 3)
            ret_x = ( (x+w/2) /self.ratio_w)
            ret_y = ( (y+h/2) /self.ratio_h)
        # show the frame to screen ...
        if args.camera_switch : cv2.imshow("apple", frame)
        # return values as point ...
        return ( ret_x, ret_y )


#######################################
#   track a banana in front of cam and return coordinates ...
    def detectBanana(self, bananaOption):
        frame = self.getRGBFrame()
        (x, y) = (ret_x, ret_y) = pointXY

        #   here start tracking ...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bananas = banana_cascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30) )

#        # apply non-maxima suppression to the bounding boxes using an overlap threshold
#        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bananas])
#        bananas = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # mark bananas with rectangle around ...
        for (x, y, w, h) in bananas:
            cv2.circle(frame, (int(x+w/2), int(y+h/2)), int(w/2), (0, 255, 0), 3)
            ret_x = ( (x+w/2) /self.ratio_w)
            ret_y = ( (y+h/2) /self.ratio_h)
        # show the frame to screen ...
        if args.camera_switch : cv2.imshow("banana", frame)
        # return values as point ...
        return ( ret_x, ret_y )


#######################################
    #   track faces in front of a webcam and return coordinates ...
    def detectFace(self, faceOption):
        frame = self.getRGBFrame()
        (x, y) = (ret_x, ret_y) = pointXY
        #   here start tracking ...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30) )
        # mark faces with rectangle around ...
        for (x, y, w, h) in faces:
            cv2.circle(frame, (int(x+w/2), int(y+h/2)), int(w/2), (0, 255, 0), 3)
            ret_x = ( (x+w/2) /self.ratio_w)
            ret_y = ( (y+h/2) /self.ratio_h)
            # detect eyes in face ROI ...
            if faceOption is not None :
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.circle(roi_color, (int(ex+ew/2), int(ey+eh/2)), int(ew/2), (255, 0, 0), 1)
        # show the frame to screen ...
        if args.camera_switch : cv2.imshow("cam", frame)
        # return values as point ...
        return ( ret_x, ret_y )


#######################################
    #   track persons in front of a webcam and return coordinates ...
    def detectPeople(self, peopleOption):
        frame = self.getRGBFrame()
        (x, y) = (ret_x, ret_y) = pointXY

        #   here start to detect people in the image
#        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=( 8,  8), scale=1.05)
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

        # apply non-maxima suppression to the bounding boxes using an overlap threshold
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # mark people with boxes ...
        for (x, y, w, h) in pick:
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)
            ret_x = ( (x+w/2) /self.ratio_w)
            ret_y = ( (y+h/2) /self.ratio_h)
        # show the frame to screen ...
        if args.camera_switch : cv2.imshow("cam", frame)
        # return values as point ...
        return ( ret_x, ret_y )


#######################################
    # destructor: cleanup the camera and close any open windows ...
    def trackQuit(self):
        self.camera.release()
        # show the frame to screen ...
        if args.camera_switch : cv2.destroyAllWindows()
#
#
##########################################################################################




#
#
##########################################################################################
#   creating a class for making face instances ...
class Face(object):

    def __init__(self, screen):
        self.pos = (0,0)
        self.screen = screen
        self.screen_w,self.screen_h = screen.get_size()
        #   create backgound
        self.bgRGB = (000, 000, 000)    # self.bgRGB = (255, 255, 255, 0)
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill( self.bgRGB )
        #   display the background
        screen.blit(background, self.pos)
        self.lastExpr = "plain"


#######################################
    #   creating a dictionary / library of images -> only loaded once ...
    def add_image(self, path):
        global imageList
        image = imageList.get(path)
        if image == None:
#            print ("Loading Image:  ", path)
            canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
            # image = pygame.image.load(canonicalized_path)
            image = pygame.image.load(canonicalized_path).convert_alpha()
            image = pygame.transform.scale(image, (self.screen_w, self.screen_h))
            imageList[path] = image
        return image


#######################################
    #   blit with alpha channel, for cases where there is alpha per pixels - in order to avoid shadows ...
    def blit_alpha(target, source, location, opacity):
        x = location[0]
        y = location[1]
        temp = pygame.Surface((source.get_width(), source.get_height())).convert()
        temp.blit(target, (-x, -y))
        temp.blit(source, self.pos)
        temp.set_alpha(opacity)
        target.blit(temp, location)


#######################################
    #   function to call and show the facial expressions ...
    def setExpression(self, expression):
        # memorise what was the last used expression, for reference ...
        if expression in exprsList :
            self.lastExpr = expression
        else :
            expression = self.lastExpr

        # fetch expression elements and create a face ...
        # first make sure to load the basis ...
        if expression not in 'blink' :
            for element in sorted(exprsList['basis'].keys()):
                #   eyes get an extra treat, in order to change direction ...
                if eyesMatch in element :
                    self.eyesFollow( exprsList['basis'][element] )
                else :
                    self.screen.blit( self.add_image( exprsList['basis'][element] ), self.pos)

        # loop through the expression elements ...
        for element in sorted(exprsList[expression].keys()):
            #   eyes get an extra treat, in order to change direction ...
            if eyesMatch in element :
                self.eyesFollow( exprsList['basis'][element] )
            else :
                self.screen.blit( self.add_image( exprsList[expression][element] ), self.pos)


#######################################
    #   timed function for blinking with eyes ...
    def timerBlnk(self):
        merke = self.lastExpr
        self.setExpression('blink')
        pygame.display.flip()
        pygame.time.wait( random.randint(eyeRangeBlnkDur[0],eyeRangeBlnkDur[1]) )
        self.setExpression(merke)
        pygame.display.flip()


#######################################
    #   timed function for squinting with eyes ... using internal timer for countdown, to still move the eyes ...
    def timerSqunt(self):
        merke = self.lastExpr
        start_ticks=pygame.time.get_ticks()
        milliseconds=(pygame.time.get_ticks()-start_ticks)
        stopp = milliseconds + random.randint(eyeRangeBlnkDur[0]*5,eyeRangeBlnkDur[1]*5)
        while milliseconds < stopp :
            milliseconds=(pygame.time.get_ticks()-start_ticks)
            ############
            # IMPORTANT: we need here an extra internal pygame event-loop, otherwise no updates, e.g. mouse positions ...
            for event in pygame.event.get(): a = 'A' # but we don't need to do anything useful here ...
            ############
            self.setExpression('squint')
            pygame.display.flip()
        self.setExpression(merke)
        pygame.display.flip()


#######################################
    #   moving eyes according to given pointer direction ...
    def eyesFollow(self, keyValue):
        pntsXY = facePointXY

#        ellipticPath = []
#        for degree in range(360):
#            x = math.cos(degree * 2*math.pi / 360) * (eyeRadiusEllpsX)
#            y = math.sin(degree * 2*math.pi / 360) * (eyeRadiusEllpsY)
#            ellipticPath.append( (x+eyeRadiusEllpsX+2, y+eyeRadiusEllpsY+2) )

        #   let's create a surface to hold our ellipse:
#        surfaceE = pygame.Surface(( eyeRadiusEllpsX*2*1.2, eyeRadiusEllpsY*2*1.2 ), pygame.SRCALPHA)
#        red = (250, 50, 50)
#        size = (0, 0, eyeRadiusEllpsX*2, eyeRadiusEllpsY*2)
#        #drawing an ellipse onto the
#        ellipse = pygame.draw.ellipse(surfaceE, red, size)
#        # surfaceE = pygame.transform.rotate(surfaceE, 45)
#        self.screen.blit( surfaceE, ( eyeOffsetXl, eyeOffsetXl ) )
#        self.screen.blit( surfaceE, ( eyeOffsetXr, eyeOffsetXr ) )
        #   visualise the elliptical pathway for eye-movements
#   offset for positioning the elliptic path for eyes
#        eyeOffsetXYl = (  77, 277 )
#        eyeOffsetXYr = ( 230, 277 )
#        surfaceP = pygame.Surface(( eyeRadiusEllpsX*2*1.2, eyeRadiusEllpsY*2*1.2 ), pygame.SRCALPHA)
#        # lines(Surface, color, closed, pointlist, width=1)
#        pygame.draw.lines(surfaceP, (0, 0, 0), True, ellipticPath, 3)
#        self.screen.blit( surfaceP, eyeOffsetXYl )
#        self.screen.blit( surfaceP, eyeOffsetXYr )

## Calculates position with parametric form, explanation: http://en.wikipedia.org/wiki/Ellipse#Parametric_form_in_canonical_position
        eyesXY = (1,0)
        cntrXY = (self.screen_w/2 +12, self.screen_h/2 -25)
        refXYm = ( int( pntsXY[0]-cntrXY[0] ), int( -1.0*(pntsXY[1]-cntrXY[1]) ) )
        angleE = 0

#   print pointer coordinates ...
#        print( "Point:  ", "(", pntsXY[0], ".", pntsXY[1], ")", "    Point-Eye-Ref:  ", "(", refXYm[0], ".", refXYm[1], ")" )

        # use angle from vector: center between eyes -> mouse pointer ...
        scalarEuvXY = ( (eyesXY[0]*refXYm[0])+(eyesXY[1]*refXYm[1]) )
        betragE_uXY = ( math.sqrt( math.pow(eyesXY[0],2)+math.pow(eyesXY[1],2)) )
        betragE_vXY = ( math.sqrt( math.pow(refXYm[0],2)+math.pow(refXYm[1],2)) )

        if ( (betragE_uXY*betragE_vXY) >0 ) :
            angleE = ( scalarEuvXY / (betragE_uXY * betragE_vXY) )
        if ( refXYm[1] > 0 ) :
            angleE = math.degrees( math.acos( angleE ) )
        else :
            angleE = 360 - math.degrees( math.acos( angleE ) )
#        print("angle in degrees:   ", angleE )
        angleE = math.radians( angleE )
#        print("angle in radians:   ", angleE )

        # percentages which scale the motions ...
        percentOfEX = ( math.fabs(refXYm[0]) / ((self.screen_w/2)/100) )/100
        percentOfEY = ( math.fabs(refXYm[1]) / ((self.screen_h/2)/100) )/100
#        print("percentOfEX:  ",percentOfEX)
#        print("percentOfEY:  ",percentOfEY)

        eyesXY = ( eyesXY[0] + (eyeRadiusEllpsX*percentOfEX) * math.cos(angleE), eyesXY[1] - (eyeRadiusEllpsY*percentOfEY) * math.sin(angleE) )
        self.screen.blit( self.add_image( keyValue ), eyesXY )
#
#
##########################################################################################


##########################################################################################
#   creating a class for making head instances ...
class Head(object):

    # @param description description of the Serial device 
    # @param interval maximum interval for sending serial messages to the device
    def __init__(self, description, interval, baudrate):
        self.pos = (0, 0)
        self.interval = interval
        self.baudrate = baudrate
        # get the port of the motor controlling device
        self.hasController = False
        ports = serial.tools.list_ports.grep(description)
        for port in ports:
            self.serial_port = port
            self.hasController = True
            break

        if self.hasController:
            print("Using \"%s\" (%s)" %(self.serial_port.description, self.serial_port.device))
            self.axis = serial.Serial(self.serial_port.device, self.baudrate)
            self.axis.write_timeout = 0
        else:
            print("No device \"%s\" found" %(description))
        # moving the head to the center position
        self.lastSerialMessage = current_milli_time()
        self.moveHeadTo(self.pos)

    def __del__(self):
        # setting the position on closing leads to a malfunction (Head turns randomly)
        # self.setPos((90,0))
        if self.hasController:
            self.axis.close()

    def close(self):
        self.setPos((0,0))
        time.sleep(1)
        self.axis.close()

    def setPos(self, positionXY):
        changed = False
        if (current_milli_time() - self.lastSerialMessage) > self.interval:
            difference = tuple(map(lambda x, y: x-y, positionXY, self.pos))
            difference_abs = np.fabs(difference[0])
            # difference_abs = np.sqrt(difference[0]**2 + difference[1]**2)
            # print("difference in X: %s; difference in Y: %s" %difference)
            # print("absolute difference: %s" %difference_abs)
            if difference_abs > differenceThreshold:
                self.pos = positionXY
                self.moveHeadTo(self.pos)

    def getcurrentPos(self):
        return self.pos
    
    def getLastMilliTime(self):
        return self.lastSerialMessage

    def moveHeadTo(self, positionXY):
        # test if controller is found and if last message was more then 'interval' milliseconds ago
        if self.hasController:
            data_string = str(positionXY[0])
            data = data_string.encode()
            # print(data)
            self.axis.flushInput()
            self.axis.flushOutput()
            self.axis.write(data)
            self.lastSerialMessage = current_milli_time()

"""
    def moveHeadToX(self, positionX):
        if self.hasController and (current_milli_time()-self.lastSerialMessage) > self.interval:
            # move x-Axis
            self.lastSerialMessage = current_milli_time()
            x_asByte = str(positionX).encode()
            # print("go to %s at %d" %(x_asByte, self.lastSerialMessage))
            self.axis.flushInput()
            self.axis.flushOutput()
            self.axis.write(x_asByte)
   
    def moveHeadToY(self, positionY):
        pass
        # is not needed yet
        # if self.hasController:
            # move y-Axis (not available yet)
            # y_asByte = str(postionY).encode()
            # print(y_asByte)
            # self.yaxis.write(y_asByte)
"""
#
#
#########################################################################################


#
#
##########################################################################################
##########################################################################################
# main-function to run the program ...
def faceIt():
    pygame.init()
    #pygame.FULLSCREEN    create a fullscreen display
    #pygame.DOUBLEBUF     recommended for HWSURFACE or OPENGL
    #pygame.HWSURFACE     hardware accelerated, only in FULLSCREEN
    #pygame.OPENGL        create an OpenGL renderable display
    #pygame.RESIZABLE     display window should be sizeable
    #pygame.NOFRAME       display window will have no border or controls
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME|pygame.HWSURFACE, 32)
    pygame.mouse.set_visible(0)
    pygame.display.set_caption('FaceIt')
    #   for info ...
    print(pygame.display.Info())

    #   little helpers
    done = False
    clock = pygame.time.Clock()
    #   set_timer(eventid, milliseconds) for eye-blink ...
    pygame.time.set_timer(USEREVENT+1, random.randint(eyeRangeBlnkTme[0], eyeRangeBlnkTme[1]) )
    #   set_timer(eventid, milliseconds) for eye-squint ...
    pygame.time.set_timer(USEREVENT+2, random.randint(eyeRangeBlnkTme[0], eyeRangeBlnkTme[1]) )

    ######
    # beg experimental
    #   set_timer(eventid, milliseconds) for tracking ...
    # pygame.time.set_timer(USEREVENT+3, 200 )
    #
    # end experimental
    ######

    #   prepare screen content and make an initial face ...
    face = Face(screen)
    face.setExpression('neutral')

    ######
    # beg experimental
    # updateRect = pygame.Rect(0, screen_h/3, screen_w, screen_h/3)
    #
    # end experimental
    #####

    #   initialize head and move to center
    head = Head(serialDeviceName, serialSendInterval, serialBaudrate)
    global facePointXY
    facePointXY = (screen_w/2, screen_h/2)
    headAngleXY = (90, 90)

    ######
    # beg experimental
    #   build a keylist from available expressions
    keyList = {}
    keyNr = 0
    for expression in sorted(exprsList) :
        keyNr += 1
        print("key: ", keyNr, "    expression: ", expression)
        keyList[keyNr] = expression
    #   counter for expression-list, to run through expressions via key up/down ...
    keyIterator = 1
    #
    # end experimental
    ######

    #   loop for running the screen and control ...
    while not done:

        for event in pygame.event.get() :
            # print(current_milli_time())
            if event.type == pygame.QUIT :
                done = True

            if event.type == USEREVENT+1 :
                face.timerBlnk()

            if event.type == USEREVENT+2 and face.lastExpr == 'unhappy' :
                face.timerSqunt()

            ######
            # beg experimental
            #
            global pointXY
            pointXY = trackerCall(trackerArgs)
            
#                if event.type == USEREVENT+3 :
#                    pointXY = trackerCall(trackerArgs)
            #
            # end experimental
            ######
            
            ######
            # beg experimental
            #
            # try to convert the mouses movement into a movement of eyes and head
            
            # global facePointXY
            
            screen_size = (screen_w, screen_h)
            
            # cursor position in percentage
            # left is -1; center is 0; right is 1
            percentage = tuple(map(lambda point,size: 2.0*point/size-1.0, pointXY, screen_size))
            
            exponent = (2, 2)
            # calculate the position of the face
            percentageFace = tuple(map(lambda perc, exp: np.sign(perc)*(1+(-1)**(exp+1)*((np.sign(perc)*perc)-1.0)**exp), percentage, exponent))
            # print("percentageFace: (%s, %s)" %percentageFace)
            facePointXY = tuple(map(lambda perc, size: (perc+1)/2*size, percentageFace, screen_size))
            # print("facePointXY: (%s, %s)" %facePointXY)
            
            # calculate the position of the head
            percentageHead = tuple(map(lambda perc, exp: np.sign(perc)*(np.sign(perc)*perc)**exp, percentage, exponent))
            # print("percentageHead: (%s, %s)" %percentageHead)
            headAngleXYList = []
            i = 0
            for key, value in sorted(headRange.items()):
                angle = int(round((percentageHead[i]+1)/2 * (value["max"]-value["min"]) + value["min"]))
                headAngleXYList.append(angle)
                i += 1
            headAngleXY = tuple(headAngleXYList)
            #print("headAngleXY: (%s, %s)" %headAngleXY)
            head.setPos(headAngleXY)
            #
            # end experimental
            ######
            
#            unused code fragment
#            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
#                is_blue = not is_blue

            pressed = pygame.key.get_pressed()

            ######
            # beg experimental
            #   scroll through expressions ...
            if pressed[pygame.K_UP] or pressed[pygame.K_RIGHT] :
                if keyIterator < keyNr: keyIterator += 1
                else: keyIterator = 1
                print("keyIterator:    ", keyIterator, "  call expression:    ", keyList[keyIterator])
                face.setExpression( keyList[keyIterator] )
            if pressed[pygame.K_DOWN] or pressed[pygame.K_LEFT] :
                if keyIterator > 1: keyIterator -= 1
                else: keyIterator = keyNr
                print("keyIterator:    ", keyIterator, "  call expression:    ", keyList[keyIterator])
                face.setExpression( keyList[keyIterator] )
            # end experimental
            ######
            

            if pressed[pygame.K_q]: done = True
            #   face basics ...
            if pressed[pygame.K_n]: face.setExpression('neutral')
            if pressed[pygame.K_w]: face.setExpression('blink')
            #   expressions ...
            if pressed[pygame.K_a]: face.setExpression('afraid')
            if pressed[pygame.K_b]: face.setExpression('bored')
            if pressed[pygame.K_e]: face.setExpression('elated')
            if pressed[pygame.K_f]: face.setExpression('frustrated')
            if pressed[pygame.K_h]: face.setExpression('happy')
            if pressed[pygame.K_r]: face.setExpression('relaxed')
            if pressed[pygame.K_s]: face.setExpression('sleepy')
            if pressed[pygame.K_o]: face.setExpression('surprised')
            if pressed[pygame.K_u]: face.setExpression('unhappy')
            else:                   face.setExpression('lstEx')

            # update the clock: tick(framerate)
            clock.tick(fps)
            # update display for all changes done in the loop ...
#            pygame.display.update(updateRect)
            pygame.display.flip()
    head.close()
# this is the end ...
##########################################################################################
##########################################################################################





#######################################
# infor print out for a given called function 'title' ...
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


#######################################
# argument parser to control the program structure...
def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("-mo", "--mouse",    required = False, action='store_true',  default=False,  dest='mouse_switch',
                    help = "Face looks at mouse pointer (default) ...")

    ap.add_argument("-m", "--motion",   required = False, action='store_true',  default=False,  dest='motion_switch',
                help = "Face looks at motions in camera ...")
    
    ap.add_argument("-a", "--apple",    required = False, action='store_true',  default=False,  dest='apple_switch',
                    help = "Face looks at an apple in camera ...")

    ap.add_argument("-ba", "--banana",  required = False, action='store_true',  default=False,  dest='banana_switch',
                    help = "Face looks at a banana in camera ...")
    
    ap.add_argument("-b", "--ball",     required = False,                       default=False,  dest='ball_switch',     nargs='?',
                    help = "Face looks at a ball in camera ...                  choices are: 'green', 'red', 'blue', 'yellow' or 'None' (default=None)")

    ap.add_argument("-f", "--face",     required = False,                       default=False,  dest='face_switch',     nargs='?',
                    help = "Face looks at detected faces in camera ...          choices: 'None' or 'closest' (default=None)")

    ap.add_argument("-p", "--people",   required = False,                       default=False,  dest='people_switch',   nargs='?',
                    help = "Face looks at detected people in camera ...         choices: 'None' or 'closest' (default=None)")

    ap.add_argument("-g", "--gesture",  required = False,                       default=False,  dest='gesture_switch',  nargs='?',
                    help = "Face looks at detected gestures in camera ...       choices: 'hand' or 'gestures' (default=hand)")

    ap.add_argument("-c", "--camID",    required = False,                       default=0,      dest='camID_switch',    nargs='?',
                    help = "Set camera ID for cam(s) ...                        choices: '0', '...', 'n' or 'kinect' or 'stereo'[two webcams] (default=0)")

    ap.add_argument("-w", "--camOn",    required = False, action='store_true',  default=False,  dest='camera_switch',
                    help = "Setting to display the WebCam input ... [ available only for tracking options ] ")

    args = ap.parse_args()
    argV = vars(args)
    return args
#
#
#######################################
# call the 'main' function when script is executed ...
if __name__ == '__main__':

    args = get_arguments()

    # init camera arguments ...
    camID = 0
    cameraFlag = False
    stereoFlag = False
    kinectFlag = False

    # init a tracker object ...
    trackerCall = None
    trackerArgs = ''
    track = None

    if args.camID_switch or args.camID_switch is None :
        if 'kinect' in args.camID_switch :
            print ("Kinect active - camera ID:    ", args.camID_switch)
            camID = int(args.camID_switch)
            kinect = Kinect()
#            kinect.showFrames()
            kinectFlag = True
            cameraFlag = False

        elif 'stereo' in args.camID_switch :
            print ("Two Cams active - camera ID:  ", args.camID_switch)
            stereoFlag = True
            cameraFlag = False
            camID = int(args.camID_switch)

        else :
            camID = int(args.camID_switch)
            print ("Camera active - camera ID:    ", camID)
            cameraFlag = True

    if args.mouse_switch :
        print ("mouse pointer active ...")
        trackerCall = pygame.mouse.get_pos
        trackerArgs = ''
        cameraFlag = False

    # in case we use camera input/output ...
    elif args.motion_switch :
        print ("motion-tracking active - tracking option:               ", args.motion_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectMotions
        trackerArgs = args.motion_switch

    elif args.apple_switch :
        print ("apple-tracking active - tracking option:                ", args.apple_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectApple
        trackerArgs = args.apple_switch

    elif args.banana_switch :
        print ("banana-tracking active - tracking option:               ", args.banana_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectBanana
        trackerArgs = args.banana_switch

    elif args.ball_switch       or args.ball_switch is None :
        print ("ball-tracking active - tracking option:                 ", args.ball_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectBall
        trackerArgs = args.ball_switch

    elif args.face_switch       or args.face_switch is None :
        print ("face-tracking active - tracking option:                 ", args.face_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectFace
        trackerArgs = args.face_switch

    elif args.people_switch     or args.people_switch is None :
        print ("people-tracking active - tracking option:               ", args.people_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectPeople
        trackerArgs = args.people_switch

    elif args.gesture_switch    or args.gesture_switch is None :
        print ("hand-tracking active - tracking option:                 ", args.gesture_switch)
        if cameraFlag : track = Track(camID)
        trackerCall = track.detectGestures
        trackerArgs = args.gesture_switch

    else :
        print ("all options default ...")
        trackerCall = pygame.mouse.get_pos
        trackerArgs = ''
        cameraFlag = False

# init a Zar-face ...
    faceIt()

#    info('main line')
#    if cameraFlag :
##        cv2.setNumThreads(0)
#        track = Track(0)
#        runInParallel(track.detectSomething(), faceIt())
#    else :
#        faceIt()

    if cameraFlag :
        track.trackQuit()
    if kinectFlag :
        kinect.stopClose()

    print ("QUIT ...")
#
#
##########################################################################################


