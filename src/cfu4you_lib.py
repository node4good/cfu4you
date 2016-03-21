import time
import webbrowser
import traceback
import math
import cv2
import numpy
import os
import sys
from logging import FileHandler
from vlogging import VisualRecord as _VisualRecord
#import scipy.ndimage as scind
import logging
import re
from pydash import py_


DILATOR_SIZE = 100
MAX_HEIGHT = 40.0
MAX_WIDTH = 10.0
ADJUSTED_HEIGHT = (MAX_HEIGHT - 1) * 1.4 - 2.0
ADJUSTED_WIDTH = (MAX_WIDTH - 1) * 8.0 - 2.0
DEFECT_MIN_SIZE = 2
RECT_SUB_PIX_PADDING = 20
K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
K_CIRC_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K_CIRC_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_CIRC_25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
K_SQ_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CROSS_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
K_CROSS_5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
ADPT_BLOCK_SIZE = 51


class VisualRecord(_VisualRecord):
    def __init__(self, title, imgs, footnotes="", fmt="png"):
        if isinstance(imgs, (list, tuple, set, frozenset)):
            multi = True
        else:
            imgs = [imgs]
            multi = False

        if max(imgs[0].shape[:2]) < 500:
            imgs = [cv2.resize(img.astype(numpy.uint8), None, fx=2, fy=2) for img in imgs]

        if multi:
            max_w = imgs[0].shape[1]
            if max_w > 1900:
                fact = 1.0 / len(imgs) * (1900.0 / max_w)
                imgs = [cv2.resize(img.astype(numpy.uint8), None, fx=fact, fy=fact) for img in imgs]
        else:
            pass
            # imgs = cv2.resize(imgs.astype(numpy.uint8), None, fx=0.5, fy=0.5)

        _VisualRecord.__init__(self, title, imgs, footnotes, fmt)


class Helper(object):
    OUTPUT_PREFIX = ''
    time_stamp = format(int((time.time() * 10) % 10000000), "6d")
    logger = logging.getLogger("cfu4you")
    htmlfile = 'cfu4you' + time_stamp + '.html'
    fh = FileHandler(htmlfile, mode="w")
    fh._old_close = fh.close

    def on_log_close(h=htmlfile, fh=fh):
        if 'last_type' in sys.__dict__:
            print '/'.join(["file:/", os.getcwd().replace('\\', '/'), h])
        else:
            webbrowser.open_new_tab(h)
        fh._old_close()

    fh.close = on_log_close
    fh.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    ts = time.time()

    def __getattribute__(self, met):
        ret = object.__getattribute__(cv2, met)
        if not hasattr(ret, '__call__') or met in ('waitKey', 'circle'):
            return ret

        def wrapped(*iargs, **ikwargs):
            t0 = time.time()
            i_ret = ret(*iargs, **ikwargs)
            t = time.time() - t0
            Helper.logText("{0:<30} [{1:4}]ms", met, int(t * 1000))
            return i_ret

        return wrapped

    @classmethod
    def log_format_message(cls, msg, *args):
        old = cls.ts
        cls.ts = time.time()
        d = cls.ts - old
        formatted_lines = traceback.format_stack()
        good_lines = filter(lambda s: 'in log' not in s, formatted_lines)
        line_no = good_lines[-1].split(',')[1][1:]
        d_msg = "[{0:4.0f}]ms - {1:<8} - {2:<60}<br>".format(d * 1000, line_no, msg.format(*args))
        return d_msg

    @classmethod
    def logText(cls, msg, *args):
        d_msg = cls.log_format_message(msg, *args)
        cls.logger.info(d_msg)

    @classmethod
    def log(cls, msg, imgs, *args):
        cls.logText(msg, args)
        cls.logger.debug(VisualRecord("", imgs))
        cls.fh.flush()

    @classmethod
    def log_pics(cls, imgs, *args):
        formatted_lines = traceback.format_stack()
        var_names = re.split('\[|\]', formatted_lines[-2])[1]
        names = re.split(', ?', var_names)
        for n, im in zip(names, imgs):
            cv2.imwrite(cls.OUTPUT_PREFIX + '_' + n + '.jpg', im)
        cls.log(var_names, imgs, *args)

    @classmethod
    def log_overlay(cls, img, mask, *args):
        formatted_lines = traceback.format_stack()
        var_names = re.split(', ?|\)', formatted_lines[-2])[3]
        b,g,r = cv2.split(img)
        mask2 = cv2.multiply(mask, 2 / 255.0, dtype=cv2.CV_32F)
        mask3 = cv2.multiply(mask, 0.6 / 255.0, dtype=cv2.CV_32F)
        mask2[mask == 0] = 1.0
        mask3[mask == 0] = 1.0
        b2, g2 = [cv2.multiply(c, mask3, dtype=cv2.CV_8U) for c in (b, g)]
        r2 = cv2.multiply(r, mask2, dtype=cv2.CV_8U)
        imgs = [cv2.merge([b2, g2, r2])]
        cls.log(var_names, imgs, *args)

""":type : cv2"""
mycv2 = Helper()


def equalizeMulti(img):
    return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])


def block(shape, block_shape):
    """Create a labels image that divides the image into blocks

    shape - the shape of the image to be blocked
    block_shape - the shape of one block

    returns a labels matrix and the indexes of all labels generated

    The idea here is to block-process an image by using SciPy label
    routines. This routine divides the image into blocks of a configurable
    dimension. The caller then calls scipy.ndimage functions to process
    each block as a labeled image. The block values can then be applied
    to the image via indexing. For instance:

    labels, indexes = block(image.shape, (60,60))
    minima = scind.minimum(image, labels, indexes)
    img2 = image - minima[labels]
    """
    np = numpy
    shape = np.array(shape)
    block_shape = np.array(block_shape)
    i, j = np.mgrid[0:shape[0], 0:shape[1]]
    ijmax = (shape.astype(float) / block_shape.astype(float)).astype(int)
    ijmax = np.maximum(ijmax, 1)
    multiplier = ijmax.astype(float) / shape.astype(float)
    i = (i * multiplier[0]).astype(int)
    j = (j * multiplier[1]).astype(int)
    labels = i * ijmax[1] + j
    indexes = np.array(range(np.product(ijmax)))
    return labels, indexes


# def preprocess(img, block_size):
#     # For background, we create a labels image using the block
#     # size and find the minimum within each block.
#     Helper.logText('preprocess 1')
#     labels, indexes = block(img.shape[:2], numpy.array((block_size, block_size)))
#     Helper.logText('preprocess 2')
#     min_block = numpy.zeros(img.shape)
#     Helper.logText('preprocess 3')
#     minima = scind.minimum(img, labels, indexes)
#     Helper.logText('preprocess 4')
#     min_block[labels != -1] = minima[labels[labels != -1]]
#     Helper.logText('preprocess 5')
#     ret = min_block.astype(numpy.uint8)
#     Helper.logText('preprocess 6')
#     return ret


def blowup(img):
    img = mycv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = mycv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    h = clahe.apply(h)
    v = clahe.apply(v)
    s = clahe.apply(cv2.multiply(s, 2))
    per = int(numpy.percentile(v, 99))
    _, threshold = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold


# def deluminate(img, rad=70):
#     krad = rad + (0 if rad % 2 else 1)
#
#     # smoothed_image = cv2.medianBlur(img, krad)
#     # img1 = mycv2.resize(smoothed_image, (img.shape[1] / rad, img.shape[0] / rad), interpolation=cv2.INTER_LINEAR)
#     # avg_image = mycv2.resize(img1, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
#     # smoothed_image2 = mycv2.medianBlur(avg_image, krad)
#     # delumi = mycv2.subtract(img, smoothed_image / 2, dtype=cv2.CV_8U)
#     # delumi[delumi < 64] = 0
#
#     smoothed_image = mycv2.medianBlur(img, krad)
#     avg_image = preprocess(smoothed_image, rad)
#     smoothed_image2 = mycv2.blur(avg_image, (krad, krad))
#     delumi = mycv2.subtract(img, smoothed_image2, dtype=cv2.CV_8U)
#     normalized = mycv2.equalizeHist(delumi)
#     Helper.log('deluminate', [img, smoothed_image, avg_image, smoothed_image2, delumi, normalized])
#     return normalized



def findROIs(img):
    min_r = min(img.shape[:2]) / 4
    circles = mycv2.HoughCircles(img, mycv2.HOUGH_GRADIENT, 64, 2 * min_r,
                                 param1=255, param2=4 * min_r,
                                 minRadius=min_r, maxRadius=3 * min_r)
    if circles is None:
        Helper.logText("no ROIs")
        return None
    Helper.logText("initial number of ROIs - {0:d}".format(len(circles)))
    filtered_circles = circles[0].astype(numpy.int16)
    r025 = filtered_circles[0][2] / 2
    filtered_circles = filter(lambda c: c[0] > r025 and c[1] > r025, filtered_circles)
    Helper.logText("number ROIs at least 1/2 r - {0:d}".format(len(filtered_circles)))
    filtered_circles = filter(lambda c: c[0] + c[2] < img.shape[1] + (c[2] / 5) and c[1] + c[2] < img.shape[0] + (c[2] / 5), filtered_circles)
    Helper.logText("number ROIs at least 80%% in - {0:d}".format(len(filtered_circles)))
    best_c = filtered_circles[0]
    filtered_circles = filter(lambda c: c[2] > (0.8 * best_c[2]), filtered_circles)
    Helper.logText("number rough ROIs {0:d}".format(len(filtered_circles)))
    circles_left2right = sorted(filtered_circles, key=lambda c: c[2], reverse=True)
    ret = [{'c': (c[0], c[1]), 'r': c[2]} for c in circles_left2right]
    return ret


def cropROI(ROI, *imgs):
    out = []
    for img in imgs:
        info = img.copy()
        mycv2.circle(info, ROI['c'], ROI['r'], (0, 255, 0), thickness=4)
        clip = mycv2.getRectSubPix(img, (2 * ROI['r'], 2 * ROI['r']), ROI['c'])
        color = 1 if len(img.shape) < 3 else (1, 1, 1)
        neg = clip.copy()
        mycv2.circle(neg, (ROI['r'], ROI['r']), ROI['r'], color, -1)
        cropped = mycv2.subtract(clip, neg)
        out.append(cropped)
    Helper.log_pics([info, clip, neg, cropped])
    return out


class ContourStats(object):
    def __init__(self, contour, M, cx, cy, offset, width, height, roundness, regularity,
                 area, rect, perimeter, hull, defects):
        self.contour = contour
        self.M = M
        self.cx = cx
        self.cy = cy
        self.offset = offset
        self.width = width
        self.height = height
        self.roundness2 = 4 * math.pi * area / (perimeter * perimeter)
        self.roundness = roundness
        self.regularity = regularity
        self.area = area
        self.rect = rect
        self.perimeter = perimeter
        self.hull = hull
        self.defects = defects
        m = map(lambda i: i[0][3] / 256.0, self.defects[1:]) if self.defects is not None else []
        self.reduced_defects = len(filter(lambda i: i > DEFECT_MIN_SIZE, m))
        self.is_size = 50 < self.area < 5000
        self.is_round = self.roundness < 1.25 and self.regularity < 0.3
        self.count = 0
        if self.is_size and self.is_round:
            self.count = 1 if self.reduced_defects < 2 else self.reduced_defects

    def __getstate__(self):
        state = self.__dict__
        del state['contour']
        del state['M']
        del state['rect']
        del state['hull']
        del state['defects']
        return state

    @classmethod
    def from_contour(cls, cnt, *args):
        if len(cnt) < 5:
            return None
        area = cv2.contourArea(cnt)
        if area < 1:
            return None
        rect = cv2.fitEllipse(cnt)
        mw = rect[1][0]
        mh = rect[1][1]
        rect_size = mw * mh
        if rect_size <= 0:
            return None

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hullPoints = cv2.convexHull(cnt, returnPoints=0)
        defects = cv2.convexityDefects(cnt, hullPoints) if (len(hullPoints) > 2) else [[]]
        area_circularity = (rect_size * math.pi) / (area * 4.0)
        roundness = abs(math.log(area_circularity, 1.1))
        rect_proportion = mw / mh
        regularity = abs(math.log(rect_proportion, 10))
        ret = cls(cnt, M=M, cx=cx, cy=cy, offset=(-x, -y), width=w, height=h, roundness=roundness,
                  regularity=regularity, area=area, rect=rect, perimeter=perimeter, hull=hull, defects=defects)
        return ret


    @classmethod
    def find_contours(cls, img, predicate=None, inner=True):

        ret = cv2.findContours(img.copy(), cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST)
        img_marked, contours, [hierarchy] = ret
        stats = py_(hierarchy)\
            .map(lambda x, i: contours[i] if (hierarchy[i][3] == -1 or inner) else [])\
            .map(cls.from_contour)\
            .compact()\
            .filter(predicate)\
            .value()
        Helper.logText("got {0} markers out of {1} raw", len(stats), len(hierarchy))
        return stats
