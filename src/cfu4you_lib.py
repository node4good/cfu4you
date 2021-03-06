import time
import webbrowser
import traceback
import math
import re
import cv2
import numpy
import json

from LogHelper import LogHelper

chrome = webbrowser.WindowsDefault()
DILATOR_SIZE = 150
PLATE_PERIF_RAD = 300
DEFECT_MIN_SIZE = 2
MAX_HEIGHT = 40.0
MAX_WIDTH = 10.0
ADJUSTED_HEIGHT = (MAX_HEIGHT - 1) * 1.4 - 2.0
ADJUSTED_WIDTH = (MAX_WIDTH - 1) * 8.0 - 2.0
RECT_SUB_PIX_PADDING = 20
K_RECT_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CIRCLE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
K_CIRCLE_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K_CIRCLE_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
K_CIRCLE_21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
ADAPTIVE_BLOCK_SIZE = 51
ALL_CONTOURS = -1
COLORMAP = [
    (255, 0, 0),        # red
    (0, 255, 0),        # green
    (0, 0, 255),        # blue
    (255, 255, 0),      # yellow
    (255, 00, 255),     # magenta
    (0, 255, 255),      # cyan
]


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ContourStats):
            return obj.__dict__
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


class Helper(LogHelper):
    CLAHE = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(10, 10))


    def __getattribute__(self, attr_name):
        """
        This is some weird OpenCV wrapper to measure timing
        @param attr_name: name of original method
        @type attr_name: string
        @return: the wrapped method
        @rtype: function
        """

        if attr_name in ('equalizeHist',):
            ret = object.__getattribute__(self, attr_name)
        else:
            ret = object.__getattribute__(cv2, attr_name)

        if not hasattr(ret, '__call__') or attr_name in ('waitKey',):
            return ret

        def wrapped(*iargs, **ikwargs):
            t0 = time.time()
            i_ret = ret(*iargs, **ikwargs)
            t = time.time() - t0
            Helper.logText("{0:<50} {1:4}-cv2", attr_name, int(t * 1000))
            return i_ret

        return wrapped


    @classmethod
    def log_pics(cls, imgs, save=False, *args):
        stack_lines = traceback.extract_stack()
        call_code = stack_lines[-2][3]
        if '[' in call_code:
            var_names = re.split('[\[\]]', call_code)[1]
            names = re.split(', ?', var_names)
        else:
            var_names = ''
            names = range(len(imgs), 1)
        if save:
            for n, im in zip(names, imgs):
                cv2.imwrite(cls.OUTPUT_PREFIX + '_' + str(n) + '.jpg', im)
        cls.log(var_names, imgs, *args)

    @classmethod
    def log_pic(cls, img, save=False, *args):
        formatted_lines = traceback.format_stack()
        var_names = re.split('[()]', formatted_lines[-2])[1]
        name = re.split(', ?', var_names)[0]
        if save:
            cv2.imwrite(cls.OUTPUT_PREFIX + '_' + name + '.jpg', img)
        cls.log(var_names, [img], *args)

    @classmethod
    def log_overlay(cls, img, mask, *args):
        formatted_lines = traceback.format_stack()
        var_names = re.split(', ?|\)', formatted_lines[-2])[3]
        b, g, r = cv2.split(img)
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


def equalizeMulti(img, is_clahe=False, chans=(0, 1, 2)):
    f1 = mycv2.CLAHE.apply if False else mycv2.equalizeHist

    def f2(i, c):
        f1(c) if i in chans else c
    return cv2.merge([f2(*p) for p in enumerate(cv2.split(img))])


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


def preprocess(img, block_size):
    import scipy.ndimage as scind

    # For background, we create a labels image using the block
    # size and find the minimum within each block.
    Helper.logText('preprocess 1')
    labels, indexes = block(img.shape[:2], numpy.array((block_size, block_size)))
    Helper.logText('preprocess 2')
    min_block = numpy.zeros(img.shape)
    Helper.logText('preprocess 3')
    minima = scind.minimum(img, labels, indexes)
    Helper.logText('preprocess 4')
    min_block[labels != -1] = minima[labels[labels != -1]]
    Helper.logText('preprocess 5')
    ret = min_block.astype(numpy.uint8)
    Helper.logText('preprocess 6')
    return ret


def blowup_threshold(img):
    v = mycv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v_eq = blowup(v)
    dilated = cv2.dilate(v_eq, K_CIRCLE_15)
    return dilated


def blowup(img):
    v1 = mycv2.equalizeHist(img)
    v2 = numpy.square(v1.astype(numpy.uint32))
    v4 = numpy.square(v2)
    v_t = numpy.right_shift(v4, 24).astype(numpy.uint8)
    per = int(numpy.percentile(v_t, 80))
    v_per = mycv2.subtract(v_t, per)
    v_eq = mycv2.equalizeHist(v_per)
    # _, t = mycv2.threshold(v_t, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # t_closed = cv2.morphologyEx(t, cv2.MORPH_OPEN, K_CIRCLE_7)
    # t_bin = t_closed/255
    # v_mask = cv2.multiply(v_t, t_bin)
    # v_eq = mycv2.equalizeHist(v_mask)
    Helper.log_pics([img, v1, v_t, v_eq])
    return v_eq


def blowup_roi(img):
    # img1 = equalizeMulti(img)
    # img2 = equalizeMulti(img1, True, [0])
    img_hsv = mycv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = mycv2.split(img_hsv)
    # v1d = deluminate(v)
    # Helper.log_pic(v1d)
    # t = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, -0.1)
    # t1 = cv2.erode(t, K_CIRCLE_15)
    # t2 = cv2.morphologyEx(t1, cv2.MORPH_CLOSE, K_CIRCLE_21)
    # ret, markers = cv2.connectedComponents(t1)
    # markers[markers == -1] = 0
    # m = cv2.watershed(img2, markers)
    # Helper.log_pic(m.astype(numpy.uint8))
    # Helper.log_pic(t1)
    # Helper.log_pic(t2)
    v_eq = blowup(v)
    Helper.log_pics([v, v_eq, s, h])
    blown_hsv = mycv2.merge([h, s, v_eq])
    blown = mycv2.cvtColor(blown_hsv, cv2.COLOR_HSV2BGR)
    return blown


def deluminate(img, rad=80):
    krad = rad + (0 if rad % 2 else 1)

    smoothed_image = cv2.medianBlur(img, krad)
    img1 = mycv2.resize(smoothed_image, (img.shape[1] / rad, img.shape[0] / rad), interpolation=cv2.INTER_LINEAR)
    avg_image = mycv2.resize(img1, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    smoothed_image2 = mycv2.medianBlur(avg_image, krad)
    delumi = mycv2.subtract(img, smoothed_image / 2, dtype=cv2.CV_8U)
    delumi[delumi < 64] = 0
    normalized = mycv2.equalizeHist(delumi)
    Helper.log_pics([img, smoothed_image, avg_image, smoothed_image2, delumi, normalized])

    # PT_BR = (img.shape[0] - 1, img.shape[1] - 1)
    # smoothed_image = mycv2.medianBlur(img, 41)
    # filled = cv2.floodFill(smoothed_image, None, (0,0), 255, loDiff=0, upDiff=8)[1]
    # filled = cv2.floodFill(filled, None, (0,img.shape[0]-1), 255, loDiff=0, upDiff=8)[1]
    # filled = cv2.floodFill(filled, None, (img.shape[0]-1,0), 255, loDiff=0, upDiff=8)[1]
    # filled = cv2.floodFill(filled, None, PT_BR, 255, loDiff=0, upDiff=5)[1]
    # smoothed_image1 = mycv2.medianBlur(filled, krad)
    # avg_image = preprocess(smoothed_image1, rad)
    # smoothed_image2 = mycv2.blur(avg_image, (krad, krad))
    # delumi = mycv2.subtract(img, smoothed_image2, dtype=cv2.CV_8U)
    # delumi[delumi < 16] = 0
    # normalized = mycv2.equalizeHist(delumi)
    # Helper.log_pics([img, smoothed_image, filled, smoothed_image1, avg_image, smoothed_image2, delumi, normalized])
    return normalized


def segment(img):
    mask = numpy.zeros(img.shape, dtype=numpy.uint8)
    r = mask.shape[0] / 2
    cv2.circle(mask, (r, r), r - PLATE_PERIF_RAD, (255, 255, 255), cv2.FILLED)
    core = numpy.bitwise_and(img, mask)
    periphery = img - core
    Helper.log_pics([core, periphery])
    return core, periphery



def is_80percent_in(shape):
    return lambda c: c[0] + c[2] < shape[1] + (c[2] / 5) and c[1] + c[2] < shape[0] + (c[2] / 5)


def findROIs(img):
    min_r = min(img.shape[:2]) / 4
    max_r = int(min_r * 2.4)
    grain = min_r / 10
    ret = mycv2.HoughCircles(img, mycv2.HOUGH_GRADIENT, grain, 2 * min_r,
                             param1=255, param2=max_r,
                             minRadius=min_r, maxRadius=max_r)
    if ret is None:
        Helper.logText("no ROIs")
        return None
    Helper.logText("initial number of ROIs - {0:d}".format(len(ret)))
    circles = ret[0].astype(numpy.int16)
    r50 = circles[0][2] / 2
    circles = filter(lambda c: c[0] > r50 and c[1] > r50, circles)
    Helper.logText("number ROIs at least 1/2 r: {0:d}".format(len(circles)))
    circles = filter(is_80percent_in(img.shape), circles)
    Helper.logText("number ROIs at least 80% in: {0:d}".format(len(circles)))
    info = mycv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for ci in circles:
        mycv2.circle(info, (ci[0], ci[1]), ci[2], (0, 255, 0), thickness=4)
    Helper.log_pic(info)
    best_c = circles[0]
    circles = filter(lambda c: c[2] > (0.8 * best_c[2]), circles)
    Helper.logText("number rough ROIs {0:d}".format(len(circles)))
    circles_left2right = sorted(circles, key=lambda c: c[2], reverse=True)
    ret = map(lambda c: {'c': (c[0], c[1]), 'r': int(c[2] + DILATOR_SIZE)}, circles_left2right)
    return ret


ROI_PAD = 2
def cropROI(ROI, *imgs):
    out = []
    for img in imgs:
        info = img.copy()
        roi_c = ROI['c']
        roi_r = ROI['r']
        mycv2.circle(info, roi_c, roi_r, (0, 255, 0), thickness=4)
        border_color = 0 if len(img.shape) < 3 else (0, 0, 0)
        img_b = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=border_color)
        cent = (roi_c[0] + 100, roi_c[1] + 100)
        rect = (2 * (roi_r + ROI_PAD), 2 * (roi_r + ROI_PAD))
        clip = mycv2.getRectSubPix(img_b, rect, cent)
        neg = clip.copy()
        mask_color = 1 if len(img.shape) < 3 else (1, 1, 1)
        roi_rect = (roi_r + ROI_PAD, roi_r + ROI_PAD)
        mycv2.circle(neg, roi_rect, roi_r, mask_color, -1)
        cropped = mycv2.subtract(clip, neg)
        out.append(cropped)
    # noinspection PyUnboundLocalVariable
    Helper.log_pics([info, clip, neg, cropped])
    return out


class ContourStats(object):
    def __init__(self, contour, M, cx, cy, offset, width, height, hierarchy, radius, roundness,
                 regularity, area, rect, perimeter, hull, defects, tag, is_internal, is_external):
        """

        @param contour:
        @type contour: numpy.ndarray
        @param M:
        @type M: Dict[str, float]
        @param cx:
        @type cx: int
        @param cy:
        @type cy: int
        @param offset:
        @type offset: Tuple[int, int]
        @param width:
        @type width: int
        @param height:
        @type height: int
        @param hierarchy:
        @type hierarchy: numpy.ndarray
        @param radius:
        @type radius: float
        @param roundness:
        @type roundness: float
        @param regularity:
        @type regularity: float
        @param area:
        @type area: float
        @param rect:
        @type rect: Tuple[Tuple[float, float], Tuple[float, float], float]
        @param perimeter:
        @type perimeter: float
        @param hull:
        @type hull: numpy.ndarray
        @param defects:
        @type defects: numpy.ndarray
        @param tag:
        @type tag: int
        @param is_internal:
        @type is_internal: numpy.bool_
        @param is_external:
        @type is_external: numpy.bool_
        """
        self.contour = contour
        self.M = M
        self.cx = cx
        self.cy = cy
        self.offset = offset
        self.width = width
        self.height = height
        self.hierarchy = hierarchy
        self.radius = radius
        self.roundness2 = 4 * math.pi * area / (perimeter * perimeter)
        self.roundness = roundness
        self.regularity = regularity
        self.area = area
        self.rect = rect
        self.perimeter = perimeter
        self.hull = hull
        self.defects = defects
        self.tag = tag
        self.is_internal = is_internal
        self.is_external = is_external
        m = map(lambda i: i[0][3] / 256.0, self.defects[1:]) if self.defects is not None else []
        self.reduced_defects = len(filter(lambda i: i > DEFECT_MIN_SIZE, m))
        self.is_size = 50 < self.area < 5000
        self.is_round = self.roundness < 1.25 and self.regularity < 0.3
        self.count = 0
        if self.is_size and self.is_round:
            self.count = 1 if self.reduced_defects < 2 else self.reduced_defects

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['contour']
        del state['M']
        del state['rect']
        del state['hull']
        del state['defects']
        del state['hierarchy']
        return state

    @classmethod
    def from_contour(cls, pair, tag, _=None):
        cnt, hierarchy = pair
        if len(cnt) < 5:
            return None
        area = cv2.contourArea(cnt)
        if area < 1:
            return None
        rect = cv2.fitEllipse(cnt)
        mw = rect[1][0]
        mh = rect[1][1]
        r = min(mw, mh) / 2
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
        is_external = hierarchy[3] == -1
        is_internal = hierarchy[2] == -1
        ret = cls(cnt, M=M, cx=cx, cy=cy, offset=(-x, -y), width=w, height=h, hierarchy=hierarchy, radius=r,
                  roundness=roundness, regularity=regularity, area=area, rect=rect, perimeter=perimeter, hull=hull,
                  defects=defects, tag=tag, is_internal=is_internal, is_external=is_external)
        return ret


    @classmethod
    def find_contours(cls, img, predicate=lambda x:x, mode=cv2.CHAIN_APPROX_NONE):
        """
        @rtype: Tuple[ContourStats]
        """
        from pydash import py_
        ret = cv2.findContours(img.copy(), cv2.RETR_TREE, mode)
        img_marked, contours, [hierarchy] = ret
        stats = py_(contours)\
            .zip(hierarchy)\
            .map(cls.from_contour)\
            .compact()\
            .filter(predicate)\
            .value()
        Helper.logText("got {0} markers out of {1} raw", len(stats), len(hierarchy))
        return stats
