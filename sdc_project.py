import cv2
import glob
import math
import numpy as np
import os

import matplotlib.image as mpimg
import xgboost as xgb

from scipy.ndimage.measurements import label
from skimage import color
from skimage.feature import hog

def color_hist(img, nbins=32, bins_range=(0, 256)):
    hist1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    hist2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    hist3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    bin_edges = hist1[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    return  hist_features

def extract_features(rgb):
    gray = color.rgb2gray(rgb)
    hog_features = hog(gray, feature_vector=True)
    lab  = color.rgb2lab(rgb)
    luv  = color.rgb2luv(rgb)
    hsv  = color.rgb2hsv(rgb)
    rgb_hist_features = color_hist(rgb)
    hsv_hist_features = color_hist(hsv)
    luv_hist_features = color_hist(luv)
    lab_hist_features = color_hist(lab)
    return np.hstack([
	hog_features,
	rgb_hist_features,
	hsv_hist_features,
	lab_hist_features,
	luv_hist_features]).astype(np.float64)

class CameraCalibrator:
    def __init__(self, shape):
        self.images = []
        self.shape  = shape

        self.obj_points = []
        self.img_points = []

        self.ret   = None
        self.mtx   = None
        self.dist  = None
        self.rvecs = None
        self.tvecs = None

    def default_object_points(self, nx, ny):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        return objp

    def load_images(self, path):
        for filename in glob.glob(path):
            self.images.append(mpimg.imread(filename))

    def match_points(self, nx, ny):
        for image in self.images:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                self.obj_points.append(self.default_object_points(nx, ny))
                self.img_points.append(corners)

    def calibrate(self):
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(self.obj_points, self.img_points, self.shape, None, None)

class VehicleDetector:
    def __init__(self, image_shape, selector, scaler, bst):
        self.prev_heatmap = np.zeros(image_shape)
        self.selector = selector
        self.scaler   = scaler
        self.bst      = bst
        self.clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def detect_cars(self, image):
        window_list1, images_list1 = self.slide_window(image, [450, None], [400, 550], ( 96,  96), (0.5, 0.5))
        window_list2, images_list2 = self.slide_window(image, [640, None], [400, 600], (128, 128), (0.5, 0.5))
        window_list3, images_list3 = self.slide_window(image, [500, 1000], [400, 500], ( 64,  64), (0.5, 0.5))

        window_list = np.concatenate((
            window_list3,
            window_list2,
            window_list1), axis=0)
        images_list = np.concatenate((
            images_list3,
            images_list2,
            images_list1), axis=0)

        images_list = self.selector.transform(images_list)
        scaled      = self.scaler.transform(images_list)
        pred        = self.bst.predict(xgb.DMatrix(scaled))

        heatmap = np.zeros(image.shape[0:2])

        for prob, pts in zip(pred, window_list):
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            if prob > 0.99:
                heatmap[y1:y2,x1:x2] += 1

        return heatmap

    def draw_labeled_bboxes(self, img, heatmap):
        labels = label(heatmap)
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, bbox[0], bbox[1], (147,112,219), 6)
        return img

    def process_image(self, orig_image, image_with_lines):
        img = np.copy(orig_image).astype(np.uint8)

        img[:,:,0] = self.clahe.apply(img[:,:,0])
        img[:,:,1] = self.clahe.apply(img[:,:,1])
        img[:,:,2] = self.clahe.apply(img[:,:,2])

        png = img.astype(np.float32)/255.
        heatmap = self.detect_cars(png)
        heatmap = cv2.addWeighted(heatmap, 1, self.prev_heatmap, 0.5, 0)
        self.prev_heatmap = np.copy(heatmap)
        heatmap = self.apply_threshold(heatmap, 3)
        return self.draw_labeled_bboxes(np.copy(image_with_lines), heatmap)

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        window_list = []
        images_list = []

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                window_list.append(((startx, starty), (endx, endy)))

                window = np.zeros(xy_window)
                window = img[starty:endy,startx:endx,:]
                window = cv2.resize(window, (64,64))
                images_list.append(extract_features(window))
        return np.array(window_list), np.array(images_list)

class LaneFinder:
    def __init__(self, image_size, clb, vd):
        self.image_size = image_size

        self.clb = clb
        self.vd  = vd

        self.src  = np.float32(((475, 548), (875, 548), (1200, 712), (250, 712)))
        self.dst  = np.float32(((350, 600), (940, 600), ( 940, 720), (350, 720)))
        self.M    = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

        self.curr_lines = [None, None]
        self.prev_lines = [None, None]
        self.prev_limit = 0

        self.stream = False

        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        self.yplot = np.array([x for x in range(self.image_size[1])])

    def add_prev_line(self, xplot):
        for i in range(int(self.image_size[1]*0.7), self.image_size[1]):
            x = xplot[i]
            y = self.yplot[i]
            self.warped[y,(x-15):(x+15)] = 1

    def detect_lines(self):
        image = self.rgb
        binary = np.zeros(image.shape[0:2])
        yline  = self.detect_yellow_line()
        wline  = self.detect_white_line()

        binary[(yline == 1) | (wline == 1)] = 1

        return binary

    def detect_white_line(self):
        image = self.rgb
        lb = np.array([120, 120, 180])
        ub = np.array([255, 255, 255])
        line = cv2.inRange(image, lb, ub)
        binary = np.zeros(image.shape[0:2])
        binary[(line > 0)] = 1

        return binary

    def detect_yellow_line(self):
        image = self.hsv()
        lb = np.array([ 0, 100, 200])
        ub = np.array([ 40, 255, 255])
        line   = cv2.inRange(image, lb, ub)
        binary = np.zeros(image.shape[0:2])
        binary[(line > 0)] = 1

        return binary

    def deviation_from_center(self, left, right, ndigits=2):
        center = self.image_size[0] / 2
        return round(left + right - 2 * center, ndigits)

    def fit_line(self, coefs, y):
        return coefs[0] + coefs[1]*y + coefs[2]*y**2

    def get_line(self, center, window = 50, line_width = 10):
        image = self.warped
        line  = np.zeros_like(image)
        min_width = 0
        max_width = image.shape[1] - 1
        for i in range(image.shape[0], 0, -1):
            row = i - 1
            sp = max(min_width, center - window)
            ep = min(max_width, center + window)
            if np.sum(image[row,sp:ep]) > 0:
                center = int(sp + np.mean(np.where(image[row,sp:ep] > 0)))
                l_side = max(min_width, center - line_width)
                r_side = min(max_width, center + line_width)
                line[row,l_side:r_side] = 1

        return line

    def get_line_centers(self):
        hist  = self.histogram()
        mid   = hist.shape[0]/2
        min_b = int(mid * 0.9)
        max_b = int(mid * 1.1)
        left  = np.argmax(hist[:min_b])
        right = np.argmax(hist[max_b:]) + max_b

        return left, right

    def get_line_coefs(self, x, y):
        cfs = np.polyfit(y, x, 2)
        return cfs[::-1]

    def get_line_curvature(self, x, y, ndigits=2):
        coefs = self.get_line_curvature_coefs(x, y)
        return round(((1 + (2*coefs[2] * y[-1] + coefs[1])**2)**1.5) / np.absolute(2*coefs[2]), ndigits)

    def get_line_curvature_coefs(self, x, y):
        cfs = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        return cfs[::-1]

    def histogram(self):
        return np.sum(self.warped[int(self.warped.shape[0]*0.7):,:], axis=0)

    def hsv(self):
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)

    def roi(self):
        image = self.detect_lines()
        mask  = np.zeros_like(image)
        cv2.fillPoly(mask, self.vertices(), 255)

        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def undist(self, origin = False):
        image = self.origin if origin else self.clahe
        return cv2.undistort(image, self.clb.mtx, self.clb.dist, None, self.clb.mtx)

    def vertices(self):
        return np.array([[
            ( 150, 700),
            ( 580, 450),
            ( 730, 450),
            (1200, 700)
        ]], dtype=np.int32)

    def warp_image(self):
        self.warped = cv2.warpPerspective(self.roi(), self.M, self.image_size)
        self.warped[self.warped > 0] = 1

    def xplot(self, center):
        line = self.get_line(center)
        x, y = self.x_y(line)
        cfs  = self.get_line_coefs(x, y)

        return self.fit_line(cfs, self.yplot).astype(np.int) #, y

    def x_y(self, line):
        x = []
        y = []
        for i in range(line.shape[0]):
            points = np.where(line[i,:])
            if len(points[0]) > 0:
                x.append(np.int(np.mean(points[0])))
                y.append(i)

        return np.array(x), np.array(y)


    def process_image(self, image):
        self.origin = np.copy(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image[:,:,0] = clahe.apply(image[:,:,0])
        image[:,:,1] = clahe.apply(image[:,:,1])
        image[:,:,2] = clahe.apply(image[:,:,2])
        self.clahe = image
        self.rgb   = self.undist()
        self.warp_image()

        if self.prev_lines[0] is not None:
            self.add_prev_line(self.prev_lines[0])

        if self.prev_lines[1] is not None:
            self.add_prev_line(self.prev_lines[1])

        ml, mr  = self.get_line_centers()

        try:
            xplot_l = self.xplot(ml)
        except:
            xplot_l = self.prev_lines[0]

        try:
            xplot_r = self.xplot(mr)
        except:
            xplot_r = self.prev_lines[1]

        self.prev_lines = [xplot_l,xplot_r]

        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_l = np.array([np.transpose(np.vstack([xplot_l[300:], self.yplot[300:]]))])
        pts_r = np.array([np.flipud(np.transpose(np.vstack([xplot_r[300:], self.yplot[300:]])))])
        pts = np.hstack((pts_l, pts_r))

        cv2.fillPoly(color_warp, np.int_([pts]), (60,179,113))
        newwarp = cv2.warpPerspective(color_warp, self.Minv, self.image_size)
        result = cv2.addWeighted(self.undist(origin=True), 1, newwarp, 0.3, 0)
        result = self.vd.process_image(self.undist(origin=True), result)

        l_curv_text = "Curvature for left line:  {} meters".format(self.get_line_curvature(xplot_l, self.yplot))
        r_curv_text = "Curvature for rigth line: {} meters".format(self.get_line_curvature(xplot_r, self.yplot))
        dev_text    = "Deviation from center: {}".format(self.deviation_from_center(ml, mr))
        cv2.putText(result, l_curv_text, (50,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(result, r_curv_text, (50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(result, dev_text, (50,95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return result
