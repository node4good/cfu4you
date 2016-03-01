import time
import webbrowser
import xlsxwriter
import traceback
import math
import io
import cv2
import numpy
import os
from logging import FileHandler
from vlogging import VisualRecord as _VisualRecord
import scipy.ndimage as scind
import logging

DILATOR_SIZE = 100


def find_colonies2(roi, roi_color, min_rad=12, max_rad=15):
    #    blurred = mycv2.blur(roi, (10,10))
    #    thr_clp = mycv2.Canny(blurred, 20, 100)
    circles = mycv2.HoughCircles(roi, mycv2.HOUGH_GRADIENT, 6, min_rad, param1=250, param2=45, minRadius=min_rad,
                                 maxRadius=max_rad)[0]
    Helper.logText('number of circles (colonies): {0}', circles.shape[0])
    circles = circles.astype(numpy.uint16)
    cols = numpy.zeros(roi.shape, dtype=numpy.int32)
    for c in circles[:200]:
        mycv2.circle(cols, (c[0], c[1]), c[2], 255, thickness=2)
    #        rr, cc = skid.circle(c[1], c[0], c[2], shape=cols.shape)
    #        cols[rr, cc] = 255
    #        cols[c[1], c[0]] = i
    # cols = mycv2.watershed(roi_color, cols)
    # cols[cols < 0] = 0
    cols = cols.astype(numpy.uint8)
    return cols


def find_colonies4(roi, roi_color, min_rad=8, max_rad=40):
    _, threshold = mycv2.threshold(roi, 1, 255, mycv2.THRESH_BINARY)
    min_tr = mycv2.erode(threshold, None, iterations=min_rad)
    max_tr = cv2.dilate(min_tr, None, iterations=min_rad)
    try:
        circles = \
        cv2.HoughCircles(max_tr, mycv2.HOUGH_GRADIENT, 1, 4 * min_rad, param1=250, param2=min_rad, minRadius=min_rad,
                         maxRadius=max_rad)[0]
    except:
        circles = []
    circles = circles.astype(numpy.uint16)
    cols = numpy.zeros(roi.shape, dtype=numpy.uint8)
    for c in circles:
        mycv2.circle(cols, (c[0], c[1]), c[2], 255, thickness=2)
    Helper.logText('number of circles (colonies): {0}', circles.shape[0])
    Helper.log('find_colonies4', [threshold, min_tr, max_tr, cols])
    return cols


def find_colonies3(img):
    # Setup SimpleBlobDetector parameters.
    params = mycv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    # Create a detector with the parameters
    detector = mycv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    keypoints = detector.detect(img)
    Helper.logText('number of keypoints {0}', len(keypoints))
    # Draw detected blobs as red circles.
    # mycv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = mycv2.drawKeypoints(img, keypoints, numpy.array([]), (0, 0, 255),
                                            mycv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    Helper.log('segment_on_dt', im_with_keypoints)
    return keypoints


class VisualRecord(_VisualRecord):
    def __init__(self, title, imgs, footnotes="", fmt="png"):
        if isinstance(imgs, (list, tuple, set, frozenset)):
            imgs = [cv2.resize(img.astype(numpy.uint8), None, fx=0.125, fy=0.125) for img in imgs]
        else:
            pass
            # imgs = cv2.resize(imgs.astype(numpy.uint8), None, fx=0.5, fy=0.5)
        _VisualRecord.__init__(self, title, imgs, footnotes, fmt)


class Helper(object):
    time_stamp = str(time.time())
    logger = logging.getLogger("cfu4you")
    htmlfile = 'cfu4you' + time_stamp + '.html'
    fh = FileHandler(htmlfile, mode="w")
    fh._old_close = fh.close
    fh.close = lambda h=htmlfile, fh=fh: (webbrowser.open_new_tab(h), fh._old_close())
    fh.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    ts = time.time()

    @classmethod
    def log(cls, msg, img):
        old = cls.ts
        cls.ts = time.time()
        d = cls.ts - old
        cls.logger.debug(VisualRecord("[%2.2f]s - " % d + msg, img, fmt="png"))

    @classmethod
    def logText(cls, msg, *args):
        old = cls.ts
        cls.ts = time.time()
        d = cls.ts - old
        cls.logger.info("[%4.0f]ms - " % (d * 1000) + msg.format(*args) + '\t\t\t\t\t\t\t\t\t<br>')

    def __getattribute__(self, meth):
        ret = object.__getattribute__(cv2, meth)
        if not hasattr(ret, '__call__') or meth in ('waitKey', 'circle'):
            return ret

        def wrapped(*iargs, **ikwargs):
            t0 = time.time()
            i_ret = ret(*iargs, **ikwargs)
            t = time.time() - t0
            formatted_lines = traceback.format_stack()
            Helper.logText("{0:<10} - call to {1:<20}  [{2:3}]ms", formatted_lines[-2].split(',')[1][1:], meth,
                           int(t * 1000))
            return i_ret

        return wrapped


""":type : cv2"""
mycv2 = Helper()


def myShow(name, img, wt=0):
    mycv2.namedWindow(name, mycv2.WINDOW_KEEPRATIO | mycv2.WINDOW_NORMAL)
    height = 800 if img.shape[1] > 800 else img.shape[1]
    width = int(800.0 * img.shape[1] / img.shape[0]) if img.shape[0] > 800 else img.shape[0]
    mycv2.resizeWindow(name, width, height)
    mycv2.imshow(name, img)
    mycv2.waitKey(wt)


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
    pre = img
    split = mycv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
    b_split = [clahe.apply(c) for c in split]
    img = cv2.merge(b_split)
    l = mycv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)[:, :, 1]
    l = deluminate(l)
    per = numpy.percentile(l, 90)
    Helper.logText("per %d" % per)
    th, mask = cv2.threshold(l, per, 255, cv2.THRESH_BINARY)
    mask2 = cv2.erode(mask, None, iterations=4)
    mask3 = cv2.dilate(mask2, None, iterations=7)
    l2 = cv2.multiply(l, mask3 / 255)
    Helper.log('blowup', [pre, img, l, mask, mask2, mask3, l2, cv2.subtract(l, l2)])
    return l2, img, mask


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
    delumi[delumi < 48] = 0
    normalized = mycv2.equalizeHist(delumi)
    Helper.log('deluminate', [img, smoothed_image, avg_image, smoothed_image2, delumi, normalized])
    return normalized


def crop(img, r, c):
    clip = mycv2.getRectSubPix(img, (2*r, 2*r), c)
    color = 1 if len(img.shape) < 3 else (1,1,1)
    neg = clip.copy()
    mycv2.circle(neg, (r, r), r, color, -1)
    cropped2 = mycv2.subtract(clip, neg)
    Helper.log("crop", [clip, neg, cropped2])
    return cropped2


def refine_circle_ROI(img, img_color, hi_cont, circle, search_extend_size=50, remove_rim=50):
    r = circle[2] + search_extend_size
    r2 = r * 2
    c = tuple(circle[:2])
    clipped = mycv2.getRectSubPix(img, (r2, r2), c)
    clip_color = mycv2.getRectSubPix(img_color, (r2, r2), c)
    clip_hi_cont = mycv2.getRectSubPix(hi_cont, (r2, r2), c)
    norm_closed = mycv2.equalizeHist(clipped)
    Helper.log("refine_circle_ROI 1", [clipped, clip_color, norm_closed])
    circles = mycv2.HoughCircles(clip_hi_cont, mycv2.HOUGH_GRADIENT, 1, r,
                                 param1=255, param2=4 * r,
                                 minRadius=circle[2] - search_extend_size, maxRadius=r + search_extend_size)
    if circles is None:
        better = (r - remove_rim, r - remove_rim, circle[2])
        Helper.logText("no better roi")
    else:
        Helper.logText("found better rois %d" % len(circles))
        better = circles[0][0].astype(numpy.uint16)
    r = better[2] - remove_rim
    c = (better[1], better[0])
    clipped_bw = crop(norm_closed, r, c)
    clip_color = crop(clip_color, r, c)
    clipped_bw = mycv2.equalizeHist(clipped_bw)
    return clipped_bw, clip_color


def findROIs(bw, color, hi_cont):
    min_r = min(bw.shape[:2]) / 4
    circles = mycv2.HoughCircles(hi_cont, mycv2.HOUGH_GRADIENT, 64, 2*min_r,
                                 param1=255, param2=4 * min_r,
                                 minRadius=min_r, maxRadius=3 * min_r)
    if circles is None:
        Helper.logText("no ROIs")
        return None
    Helper.logText("initial number of ROIs %d" % len(circles))
    filtered_circles = circles[0].astype(numpy.int16)
    r025 = filtered_circles[0][2] / 2
    filtered_circles = filter(lambda c: c[0] > r025 and c[1] > r025, filtered_circles)
    Helper.logText("number ROIs at least 1/4 in %d" % len(filtered_circles))
    best_c = filtered_circles[0]
    filtered_circles = filter(lambda c: c[2] > (0.8 * best_c[2]), filtered_circles)
    Helper.logText("number rough ROIs %d" % len(filtered_circles))
    circles_left2right = sorted(filtered_circles, key=lambda c: c[2], reverse=True)
    better = [refine_circle_ROI(bw, color, hi_cont, c) for c in circles_left2right]
    info = color.copy()
    mycv2.circle(info, (best_c[0], best_c[1]), best_c[2], (0, 255, 0), thickness=2)
    Helper.log("best ROI", [info])
    return better


def find_colonies1(roi, roi_color):
    global FILE_NAME
    _, threshold = mycv2.threshold(roi, 1, 255, mycv2.THRESH_BINARY)
    morphology = mycv2.morphologyEx(threshold, mycv2.MORPH_OPEN, numpy.ones((3, 3), dtype=int))
    border = mycv2.erode(morphology, None, iterations=3)
    dt = mycv2.distanceTransform(border, 2, 3).astype(numpy.uint8)
    dt = mycv2.equalizeHist(dt)
    _, dt_trash = mycv2.threshold(dt, 100, 255, mycv2.THRESH_BINARY)
    __, markers, stats, centroids = mycv2.connectedComponentsWithStats(dt_trash, connectivity=4)
    stats = [list(s) + [(s[2] * s[3] * math.pi) / (s[4] * 4.0)] for s in stats[1:]]
    centroids = centroids[1:]
    Helper.log('distanceTransform', [morphology, border, dt, dt_trash, markers])
    return markers, stats, centroids


def churn(roi, markers, stats, centroids):
    stats = [{
        'i': i,
        'x': c[0],
        'y': c[1],
        'width': s[2],
        'height': s[3],
        'area': s[4],
        'roundness': s[5],
        'size': (s[4] > 80),
        'round': (abs(s[5] - 1) < 0.1)
    } for c, s, i in zip(centroids, stats, range(1, len(stats) + 1))]
    count = reduce(lambda a, s: a + (s['size'] and s['round']), stats, 0)
    red = numpy.zeros(roi.shape[:2], dtype=numpy.uint8)
    green = numpy.zeros(roi.shape[:2], dtype=numpy.uint8)
    for s in stats:
        tag = green if s['size'] and s['round'] else red
        tag[markers == s['i']] = 255
    Helper.logText("made tags")

    chans = mycv2.split(roi)
    chans = [mycv2.subtract(chans[0], green + red), mycv2.subtract(chans[1], -green + red), mycv2.subtract(chans[2], green + -red)]
    roi_marked = mycv2.merge(chans)

    xlsx_ = os.path.join(OUTPUT_DIR, '.'.join([FILE_NAME, "_%.0f"%(time.time() * 10 % 10000), "[%d]" % count, 'xlsx']))
    workbook = xlsxwriter.Workbook(xlsx_)
    workbook.default_format_properties['valign'] = 'top'
    worksheet = workbook.add_worksheet()
    xfmt = workbook.add_format()
    xfmt_red = workbook.add_format()
    xfmt_red.set_font_color('red')
    xfmt_megenta = workbook.add_format()
    xfmt_megenta.set_font_color('magenta')
    MAX_HEIGHT = 80
    MAX_WIDTH = 20
    worksheet.set_column(8, 9, MAX_WIDTH + 4)
    Helper.logText("prepare excel")

    for s in stats:
        slc = cv2.getRectSubPix(roi_color, (s['width'] + 10, s['height'] + 10), (s['x'], s['y']))
        slcm = cv2.getRectSubPix(roi_marked, (s['width'] + 10, s['height'] + 10), (s['x'], s['y']))
        _, buf1 = cv2.imencode('.jpg', slc)
        _, buf2 = cv2.imencode('.jpg', slcm)
        image_data1 = io.BytesIO(buf1)
        image_data2 = io.BytesIO(buf2)
        image_arg1 = {'image_data': image_data1, 'positioning': 1}
        image_arg2 = {'image_data': image_data2, 'positioning': 1}
        if s['height'] > MAX_HEIGHT or s['width'] / 8 > MAX_WIDTH:
            fact_h = MAX_HEIGHT / (s['height'] + 10.0)
            fact_w = (MAX_WIDTH * 8) / (s['width'] + 10.0)
            image_arg1['x_scale'] = image_arg1['y_scale'] = image_arg2['x_scale'] = image_arg2['y_scale'] = min(fact_h, fact_w)
        worksheet.insert_image(s['i'], 8, str(s['i']), image_arg1)
        worksheet.insert_image(s['i'], 9, str(s['i']), image_arg2)
        fmt = xfmt if s['round'] else xfmt_megenta
        fmt = fmt if s['size'] else xfmt_red
        worksheet.set_row(s['i'], MAX_HEIGHT + 4, fmt)
        worksheet.write_number(s['i'], 0, s['x'])
        worksheet.write_number(s['i'], 1, s['y'])
        worksheet.write_number(s['i'], 2, s['width'])
        worksheet.write_number(s['i'], 3, s['height'])
        worksheet.write_number(s['i'], 4, s['area'])
        worksheet.write_number(s['i'], 5, s['roundness'])
        worksheet.write_boolean(s['i'], 6, s['size'])
        worksheet.write_boolean(s['i'], 7, s['round'])

    worksheet.add_table(0, 0, len(stats), 9, {
        'banded_rows': False,
        'style': 'Table Style Light 11',
        'columns': [
            {'header': 'x'},
            {'header': 'y'},
            {'header': 'width'},
            {'header': 'height'},
            {'header': 'area'},
            {'header': 'roundness'},
            {'header': 'size'},
            {'header': 'round'},
            {'header': 'pic raw'},
            {'header': 'pic marked'}
        ]
    })
    Helper.logText("imported table excel")

    _, buf_mark = cv2.imencode('.jpg', roi_marked)
    image_data_mark = io.BytesIO(buf_mark)
    worksheet2 = workbook.add_worksheet()
    worksheet2.insert_image(0, 0, 'marked', {'image_data': image_data_mark, 'positioning': 2})
    worksheet2.set_zoom(50)

    _, buf_raw = cv2.imencode('.jpg', roi)
    image_data_raw = io.BytesIO(buf_raw)
    worksheet3 = workbook.add_worksheet()
    worksheet3.insert_image(0, 0, 'raw', {'image_data': image_data_raw, 'positioning': 2})
    worksheet3.set_zoom(50)

    workbook.close()
    Helper.logText("saved excel")

    # markers_ws = mycv2.watershed(roi_color, markers)
    Helper.logText("number = %d" % count)
    # cols = cv2.watershed(roi_color, dt_trash)
    return roi_marked



OUTPUT_DIR = r"C:\code\6broad\colony-profile\output\cfu4good\RB"
# PATH_NAME = r"V:\2\Camera\5\IMG_20151125_182927.jpg" # single plate (on stand)
# PATH_NAME = r"V:\2\Camera\5\IMG_20151125_183446.jpg"  # two plates (left one cut)
# PATH_NAME = r"V:\2\Camera\1\IMG_20151104_165813.jpg"  # tiny plate
# PATH_NAME = r"V:\2\Camera\2\IMG_20151104_171622.jpg"  # tiny plate
# PATH_NAME = r"V:\2\Camera\2\IMG_20151104_171616.jpg"  # small plate
# PATH_NAME = r"V:\2\Camera\4\IMG_20151105_142456_data.jpg"
# PATH_NAME = r"c:\Users\refael\Downloads\2015-11-29.jpg"  # small cut plate
# PATH_NAME = r"C:\code\6broad\colony-profile\c4\IMG_2670.JPG"  # Lizi
# PATH_NAME = r"C:\code\colony-profile\c4\IMG_2942.JPG"  # Christina
# PATH_NAME = r"C:\code\colony-profile\c4\2015-12-11.png"  # 2015-12-11
# PATH_NAME = r"C:\code\colony-profile\c4\IMG_0649.JPG"  # insane
# PATH_NAME = r"C:\code\colony-profile\c4\2015-12-18 14.50.04.jpg"  # 72 lawn
# PATH_NAME = r"V:\2\CFU\RB\images_for_Refael\E072_d8.JPG" # Z fresh
# PATH_NAME = r"C:\code\6broad\.data\JL\IMG_20160220_130925_data.jpg"  # Z fresh

# IMGs = [
# r"V:\2\Camera\5\IMG_20151125_182156.jpg",
# r"V:\2\Camera\5\IMG_20151125_182207.jpg",
# r"V:\2\Camera\5\IMG_20151125_182218.jpg",
# r"V:\2\Camera\5\IMG_20151125_182228.jpg",
# ]
DIR_NAME = r"V:\2\CFU\RB\images_for_Refael"
for f in os.listdir(DIR_NAME):
    Helper.logText("*************** %s ***************" % f)
    PATH_NAME = os.path.join(DIR_NAME, f)
    FILE_NAME, _ext = os.path.splitext(f)
    orig = mycv2.imread(PATH_NAME)
    if orig.shape[0] < orig.shape[1]:
        orig = mycv2.transpose(orig)
    blown_bw, blown_color, hi_cont = blowup(orig)
    Helper.log('blowup', [blown_bw, blown_color, hi_cont])
    rois = findROIs(blown_bw, blown_color, hi_cont)
    roi_bw, roi_color = rois[0]
    Helper.log('found ROI', [roi_color, roi_bw])

    mrk, st, cs = find_colonies1(roi_bw, roi_color)
    colonies1_merged = churn(roi_color, mrk, st, cs)
    Helper.log(PATH_NAME, colonies1_merged)

