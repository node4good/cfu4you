import time
import webbrowser
import xlsxwriter
import traceback
import math
import io
import cv2
import numpy
import os
import sys
from logging import FileHandler
from vlogging import VisualRecord as _VisualRecord
import scipy.ndimage as scind
import logging
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

class VisualRecord(_VisualRecord):
    def __init__(self, title, imgs, footnotes="", fmt="webp"):
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
        if not hasattr(ret, '__call__') or met in ('waitKey', 'circle', 'equalizeMulti'):
            return ret

        def wrapped(*iargs, **ikwargs):
            t0 = time.time()
            i_ret = ret(*iargs, **ikwargs)
            t = time.time() - t0
            formatted_lines = traceback.format_stack()
            Helper.logText("{0:<10} - call to {1:<20}  [{2:3}]ms", formatted_lines[-2].split(',')[1][1:], met,
                           int(t * 1000))
            return i_ret

        return wrapped

    @classmethod
    def format_message(cls, msg, args):
        old = cls.ts
        cls.ts = time.time()
        d = cls.ts - old
        d_msg = "[{0:4.0f}]ms - ".format(d * 1000) + msg.format(*args) + '\t\t\t\t\t\t\t\t\t<br>'
        return d_msg

    @classmethod
    def logText(cls, msg, *args):
        d_msg = cls.format_message(msg, args)
        cls.logger.info(d_msg)

    @classmethod
    def log(cls, msg, imgs, *args):
        cls.logText(msg, args)
        cls.logger.debug(VisualRecord("", imgs))
        cls.fh.flush()

    @classmethod
    def equalizeMulti(cls, img):
        return cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])


""":type : cv2"""
mycv2 = Helper()


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


def blowup(img):
    img = mycv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = mycv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    h = clahe.apply(h)
    v = clahe.apply(v)
    s = clahe.apply(cv2.multiply(s, 2))
    img = mycv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)
    per = int(numpy.percentile(v, 90))
    Helper.logText("per {0:d}".format(per))
    ADPT_BLOCK_SIZE = 51
    threshold = mycv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADPT_BLOCK_SIZE, per/255.0)
    eroded = mycv2.erode(threshold, K_CIRC_5, iterations=2)
    dilated = mycv2.dilate(eroded, K_CIRC_5, iterations=2)
    masked = mycv2.multiply(v, dilated / 255)
    masked_eq = cv2.equalizeHist(masked)
    d = mycv2.subtract(img, cv2.merge([masked, masked, masked]))
    Helper.log('d', [d])
    Helper.log('v, threshold, eroded, dilated', [v, threshold, eroded, dilated])
    Helper.log('img, masked, threshold', [img, masked, eroded])
    return img, masked_eq, eroded


def deluminate(img, rad=70):
    krad = rad + (0 if rad % 2 else 1)

    # smoothed_image = cv2.medianBlur(img, krad)
    # img1 = mycv2.resize(smoothed_image, (img.shape[1] / rad, img.shape[0] / rad), interpolation=cv2.INTER_LINEAR)
    # avg_image = mycv2.resize(img1, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # smoothed_image2 = mycv2.medianBlur(avg_image, krad)
    # delumi = mycv2.subtract(img, smoothed_image / 2, dtype=cv2.CV_8U)
    # delumi[delumi < 64] = 0

    smoothed_image = mycv2.medianBlur(img, krad)
    avg_image = preprocess(smoothed_image, rad)
    smoothed_image2 = mycv2.blur(avg_image, (krad, krad))
    delumi = mycv2.subtract(img, smoothed_image2, dtype=cv2.CV_8U)
    normalized = mycv2.equalizeHist(delumi)
    Helper.log('deluminate', [img, smoothed_image, avg_image, smoothed_image2, delumi, normalized])
    return normalized



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
    Helper.log("crop", [info, clip, neg, cropped])
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


def find_colonies1(masked, color):
    stats = ContourStats.find_contours(masked, lambda r: (r.roundness < 4) and (4 < r.area < 16000))
    markers = numpy.zeros(masked.shape, dtype=numpy.int32)
    for i, s in enumerate(stats, 1):
        cv2.drawMarker(markers, (s.cx, s.cy), i, markerType=cv2.MARKER_SQUARE, markerSize=2)

    markers = mycv2.watershed(color, markers)
    masked2 = masked.copy()
    masked2[markers == -1] = 0
    _, mask_th = cv2.threshold(masked2, 1, 255, cv2.THRESH_BINARY)
    stats = ContourStats.find_contours(mask_th, lambda r: (r.roundness < 4) and (4 < r.area < 16000))
    stats.sort(key=lambda r: r.cy)
    contours = numpy.zeros(markers.shape, dtype=numpy.uint8)
    for i, s in enumerate(stats):
        cv2.drawContours(contours, [s.contour], -1, 7 * ((i % 64) + 1), thickness=cv2.FILLED, lineType=cv2.LINE_8)
        cv2.drawContours(contours, [s.contour], -1, 255, thickness=1, lineType=cv2.LINE_8)
    color2 = color.copy()
    color2[markers == -1] = (0, 0, 255)
    Helper.log('markers', cv2.merge([masked2, mask_th, contours]))
    Helper.log('contours', [color2])
    return stats


def churn(roi, stats, filename):
    img_eq = Helper.equalizeMulti(roi)
    roi_marked = img_eq.copy()

    workbook = xlsxwriter.Workbook(filename)
    workbook.default_format_properties['valign'] = 'top'
    worksheet = workbook.add_worksheet()
    worksheet.set_zoom(200)
    xfmt = workbook.add_format()
    xfmt_red = workbook.add_format()
    xfmt_red.set_font_color('red')
    xfmt_megenta = workbook.add_format()
    xfmt_megenta.set_font_color('magenta')
    xfmt_bold = workbook.add_format()
    xfmt_bold.set_font_color('green')
    xfmt_bold.set_bold(True)
    worksheet.set_column(7, 8, MAX_WIDTH)
    Helper.logText("prepare excel")

    cp = roi.copy()
    cv2.drawContours(cp, map(lambda s:s.contour, stats), -1, (0, 255, 0))
    cv2.imwrite(OUTPUT_DIR + '\\markers2.png', cp)

    red2 = []
    green2 = []
    """s:type : CntStats"""
    for i, s in enumerate(stats, 1):
        if s.count:
            green2.append(s.contour)
        else:
            red2.append(s.contour)
        label_point = (
            s.cx - (int(math.log(i, 10)) + 1) * 8,
            s.cy - (s.height / 2) - 2
        )
        cv2.putText(roi_marked, str(i), label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), thickness=2)
        slc = cv2.getRectSubPix(img_eq, (s.width + RECT_SUB_PIX_PADDING, s.height + RECT_SUB_PIX_PADDING), (s.cx, s.cy))
        slc_m = slc.copy()
        one_side_padding = RECT_SUB_PIX_PADDING / 2
        offset = (s.offset[0] + one_side_padding, s.offset[1] + one_side_padding)
        cv2.drawContours(slc_m, [s.contour], -1, (255, 0, 0), thickness=1, offset=offset)
        __, buf1 = cv2.imencode('.jpg', slc)
        ___, buf2 = cv2.imencode('.jpg', slc_m)
        image_data1 = io.BytesIO(buf1)
        image_data2 = io.BytesIO(buf2)
        image_arg1 = {'image_data': image_data1, 'positioning': 1}
        image_arg2 = {'image_data': image_data2, 'positioning': 1}
        a_height = s.height + RECT_SUB_PIX_PADDING
        a_width = s.width + RECT_SUB_PIX_PADDING
        if ADJUSTED_HEIGHT < a_height or ADJUSTED_WIDTH < a_width:
            fact_h = (ADJUSTED_HEIGHT) / (a_height + 16)
            fact_w = (ADJUSTED_WIDTH) / (a_width + 4)
            image_arg1['x_scale'] = image_arg1['y_scale'] = image_arg2['x_scale'] = image_arg2['y_scale'] = min(fact_h, fact_w)
        worksheet.insert_image(i, 7, str(i), image_arg1)
        worksheet.insert_image(i, 8, str(i), image_arg2)
        fmt = xfmt if s.is_size else xfmt_megenta
        fmt = fmt if s.is_round else xfmt_red
        worksheet.set_row(i, MAX_HEIGHT, fmt)
        worksheet.write_number(i, 0, i)
        worksheet.write_number(i, 1, s.cx)
        worksheet.write_number(i, 2, s.cy)
        worksheet.write_number(i, 3, s.width)
        worksheet.write_number(i, 4, s.height)
        worksheet.write_number(i, 5, s.area)
        worksheet.write_number(i, 6, s.perimeter)
        # 6 - raw
        # 7 - Marked
        worksheet.write_number(i, 9, s.count, xfmt_bold)
        worksheet.write_number(i, 10, s.roundness)
        worksheet.write_number(i, 11, s.regularity)
        worksheet.write_number(i, 12, s.reduced_defects)
        worksheet.write_boolean(i, 13, s.is_round)
        worksheet.write_boolean(i, 14, s.is_size)

    worksheet.add_table(0, 0, len(stats)+1, 14, {
        'total_row': True,
        'banded_rows': False,
        'style': 'Table Style Light 11',
        'columns': [
            {'header': 'tag No.'},
            {'header': 'x'},
            {'header': 'y'},
            {'header': 'width'},
            {'header': 'height'},
            {'header': 'area'},
            {'header': 'perimeter'},
            {'header': 'pic raw'},
            {'header': 'pic marked'},
            {'header': 'COUNT', 'total_function': 'sum'},
            {'header': 'roundness'},
            {'header': 'regularity'},
            {'header': 'defects No.'},
            {'header': 'is round'},
            {'header': 'is sized'},
        ]
    })
    Helper.logText("imported table excel")

    cv2.drawContours(roi_marked, red2, -1, (0, 0, 255))
    Helper.logText("draw red")
    cv2.drawContours(roi_marked, green2, -1, (0, 255, 0))
    Helper.logText("draw green")

    _, buf_mark = cv2.imencode('.jpg', roi_marked)
    image_data_mark = io.BytesIO(buf_mark)
    worksheet2 = workbook.add_worksheet()
    worksheet2.insert_image(0, 0, 'marked', {'image_data': image_data_mark, 'positioning': 2})
    worksheet2.set_zoom(50)
    Helper.logText("encoded marked")

    _, buf_raw = cv2.imencode('.jpg', roi)
    image_data_raw = io.BytesIO(buf_raw)
    worksheet3 = workbook.add_worksheet()
    worksheet3.insert_image(0, 0, 'raw', {'image_data': image_data_raw, 'positioning': 2})
    worksheet3.set_zoom(50)
    Helper.logText("encoded raw")

    workbook.close()
    Helper.logText(u"saved excel {0:s}".format(filename))

    # markers_ws = mycv2.watershed(roi_color, markers)
    Helper.logText(u"churn: raw {0:d}\t\tfinal {1:d}".format(len(stats), len(green2)))
    # cols = cv2.watershed(roi_color, dt_trash)

    return roi_marked


def process_file(file):
    Helper.logText("*************** {0:s} ***************".format(file))
    PATH_NAME = os.path.join(DIR_NAME, file)
    FILE_NAME, _ext = os.path.splitext(file)
    new_name = '.'.join([FILE_NAME, "_{0:4d}".format(int(time.time()) % 10000), 'xlsx'])
    filename = os.path.join(OUTPUT_DIR, new_name)
    orig = mycv2.imread(PATH_NAME)
    if orig.shape[0] < orig.shape[1]:
        orig = numpy.rot90(orig, 3)
        Helper.logText("rotated")
    blown, masked, big_blobs = blowup(orig)
    rois = findROIs(big_blobs)
    roi = rois[0]
    roi_masked, roi_color  = cropROI(roi, masked, blown)
    # roi2 = cv2.getRectSubPix(roi_color, (100, 100), (1026, 1255))
    # roihsl = cv2.cvtColor(roi2, cv2.COLOR_BGR2HLS_FULL)
    # split = cv2.split(roihsl)
    # for sat in split:
    #     sat1 = Helper.equalizeMulti(sat)
    #     slices = [cv2.threshold(sat1, i, 255, cv2.THRESH_BINARY)[1] for i in range(192, 256)]
    #     Helper.log("roi2, sat, sat1...", [roi2, sat, sat1] + slices)
    # roi_masked = masked
    # roi_color = orig
    st = find_colonies1(roi_masked, roi_color)
    # data = [s.__getstate__() for s in st]
    # df = pandas.DataFrame(data)
    # df.to_csv(filename + '.csv')
    colonies1_merged = churn(roi_color, st, filename)
    Helper.log(PATH_NAME, colonies1_merged)




# PATH_NAME = r"V:\Camera\5\IMG_20151125_182927.jpg" # single plate (on stand)
# PATH_NAME = r"V:\Camera\5\IMG_20151125_183446.jpg"  # two plates (left one cut)
# PATH_NAME = r"V:\Camera\1\IMG_20151104_165813.jpg"  # tiny plate
# PATH_NAME = r"V:\Camera\2\IMG_20151104_171622.jpg"  # tiny plate
# PATH_NAME = r"V:\Camera\2\IMG_20151104_171616.jpg"  # small plate
# PATH_NAME = r"V:\Camera\4\IMG_20151105_142456_data.jpg"
# PATH_NAME = r"c:\Users\refael\Downloads\2015-11-29.jpg"  # small cut plate
# PATH_NAME = r"C:\code\6broad\colony-profile\c4\IMG_2670.JPG"  # Lizi
# PATH_NAME = r"C:\code\colony-profile\c4\IMG_2942.JPG"  # Christina
# PATH_NAME = r"C:\code\colony-profile\c4\2015-12-11.png"  # 2015-12-11
# PATH_NAME = r"C:\code\colony-profile\c4\IMG_0649.JPG"  # insane
# PATH_NAME = r"C:\code\colony-profile\c4\2015-12-18 14.50.04.jpg"  # 72 lawn
# PATH_NAME = r"V:\CFU\RB\images_for_Refael\E072_d8.JPG" # Z fresh
# PATH_NAME = r"C:\code\6broad\.data\JL\IMG_20160220_130925_data.jpg"  # Z fresh

# IMGs = [
# r"V:\Camera\5\IMG_20151125_182156.jpg",
# r"V:\Camera\5\IMG_20151125_182207.jpg",
# r"V:\Camera\5\IMG_20151125_182218.jpg",
# r"V:\Camera\5\IMG_20151125_182228.jpg",
# ]


# DIR_NAME = r"V:\CFU\pics"
# files = ["20160308_182258_c.jpg"]

# DIR_NAME = r"V:\CFU\5"
# files = ["IMG_20151125_183850.png"]
#
DIR_NAME = r"V:\CFU\RB\images_for_Refael"
# files = os.listdir(DIR_NAME)
# random.shuffle(files)
# files = ['E072_g7.JPG'] # HARD
files = ['E072_d7.JPG'] # EASY

OUTPUT_DIR = r"C:\code\6broad\colony-profile\output\cfu4good\RB"


for f1 in files:
    process_file(f1)

