import time
import webbrowser

import cv2
import numpy
import skimage.draw as skid
from logging import FileHandler
from vlogging import VisualRecord as _VisualRecord
import scipy.ndimage as scind
import logging

DILATOR_SIZE = 100


class VisualRecord(_VisualRecord):
    def __init__(self, title, imgs, footnotes="", fmt="png"):
        if not isinstance(imgs, (list, tuple, set, frozenset)):
            imgs = [imgs]
        if max(imgs[0].shape[:2]) > 1000:
            imgs = [cv2.resize(img.astype(numpy.uint8), None, fx=0.2, fy=0.2) for img in imgs]
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    @classmethod
    def log(cls, msg, img):
        cls.logger.debug(VisualRecord(msg, img, fmt="png"))

    @classmethod
    def logText(cls, msg, *args):
        cls.logger.info(msg.format(*args) + '\t\t\t<br>')

    def __getattribute__(self, meth):
        ret = object.__getattribute__(cv2, meth)
        if not hasattr(ret, '__call__') or meth in ('waitKey', 'circle'):
            return ret

        def wrapped(*iargs, **ikwargs):
            t0 = time.time()
            i_ret = ret(*iargs, **ikwargs)
            t = time.time() - t0
            Helper.logText("call to cv2.{0} \t\t [ {1} ]ms", meth, int(t * 1000))
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

def normalized(img, min_c=10, max_c=90):
    median = numpy.percentile(img, 50)
    img_gry = mycv2.subtract(img, median)
    norm = mycv2.equalizeHist(img_gry)
    return norm


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
    i,j = np.mgrid[0:shape[0],0:shape[1]]
    ijmax = (shape.astype(float)/block_shape.astype(float)).astype(int)
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
    labels, indexes = block(img.shape[:2], numpy.array((block_size, block_size)))
    min_block = numpy.zeros(img.shape)
    minima = scind.minimum(img, labels, indexes)
    min_block[labels != -1] = minima[labels[labels != -1]]
    return min_block.astype(numpy.uint8)


def deluminate_old(img, sd=3, rad=30):
    import cellprofiler.modules.correctilluminationcalculate as clc
    import cellprofiler.cpimage  as cpi
    w = clc.CorrectIlluminationCalculate()
#    w.intensity_choice.value = clc.IC_BACKGROUND
    w.intensity_choice.value = clc.IC_REGULAR
    w.block_size.value = 100
    w.dilate_objects.value = True
    w.object_dilation_radius.value = 10
    w.smoothing_method.value = clc.SM_GAUSSIAN_FILTER
    f_img = img / 255.0
    orig_image = cpi.Image(f_img)
    avg_image = w.preprocess_image_for_averaging(orig_image)
    Helper.log('avg_image', avg_image.pixel_data)
    dilated_image = w.apply_dilation(avg_image, orig_image)
    Helper.log('dilated_image', dilated_image.pixel_data)
    smoothed_image = w.apply_smoothing(dilated_image, orig_image)
    Helper.log('smoothed_image', smoothed_image.pixel_data)

    # kernel = mycv2.getStructuringElement(mycv2.MORPH_ELLIPSE,(10,10))
    # smoothed_image = mycv2.dilate(img, kernel)
    # lumi = (mycv2.multiply(smoothed_image.astype(numpy.uint16), 0.75)).astype(numpy.uint8)
    # img_gry = mycv2.subtract(img, lumi)
    # norm = mycv2.equalizeHist(img_gry)
    # Helper.log('deluminate', (smoothed_image.astype(numpy.uint8), img_gry, norm))
    lumi = mycv2.multiply(smoothed_image.pixel_data, 255).astype(numpy.uint8)
    Helper.log('lumi', lumi)
    delumi = mycv2.subtract(img, lumi, dtype=mycv2.CV_8U)
    Helper.log('delumi', delumi)
    normalized = mycv2.equalizeHist(delumi)
    Helper.log('normalized', normalized)
    trashed = normalized[normalized <= 200] = 0
    Helper.log('trashed', normalized)
    return normalized


def deluminate(img, sd=3, rad=30):
    ker = (rad / 2) * 2 + 1
    smoothed_image = mycv2.blur(img, (ker, ker))
    Helper.log('smoothed_image', smoothed_image)
    avg_image = preprocess(smoothed_image, rad)
    Helper.log('avg_image', avg_image)
    smoothed_image = mycv2.blur(avg_image, (ker, ker))
    Helper.log('smoothed_image', smoothed_image)
    delumi = mycv2.subtract(img, smoothed_image, dtype=mycv2.CV_8U)
    Helper.log('delumi', delumi)
    delumi[delumi < 48] = 0
    Helper.log('delumi - 64', delumi)
    normalized = mycv2.equalizeHist(delumi)
    Helper.log('normalized', normalized)
    # normalized[normalized <= 200] = 0
    # Helper.log('trashed', normalized)
    return normalized


def crop(img, indexes, just_circle=True):
    if (not just_circle):
        cropped = img[min(indexes[0]):max(indexes[0]), min(indexes[1]):max(indexes[1])]
        return cropped
    clipped2 = numpy.zeros(img.shape, dtype=numpy.uint8)
    mask = numpy.zeros(img.shape, dtype=numpy.bool)
    mask[indexes] = True
    numpy.copyto(clipped2, img, where=mask)
    clipped2 = clipped2[min(indexes[0]):max(indexes[0]), min(indexes[1]):max(indexes[1])]
    return clipped2


def refine_circle_ROI(img, img_color, circle, search_extend_size=50, remove_rim=150):
    r = circle[2]
    clp_x0 = max(circle[0] - r - search_extend_size, 0)
    clp_x1 = min(circle[0] + r + search_extend_size, img.shape[1] - 1)
    clp_y0 = max(circle[1] - r - search_extend_size, 0)
    clp_y1 = min(circle[1] + r + search_extend_size, img.shape[0] - 1)
    clipped = img[clp_y0:clp_y1, clp_x0:clp_x1]
    norm = mycv2.equalizeHist(clipped)
    Helper.log('normalized rough ROI %d' % circle[0], clipped)
    kernel = mycv2.getStructuringElement(mycv2.MORPH_ELLIPSE,(3,3))
#    norm_closed = mycv2.morphologyEx(norm, mycv2.MORPH_CLOSE, kernel)
    norm_closed = norm
    Helper.log('norm_closed rough ROI %d' % circle[0], clipped)
    clip_color = img_color[clp_y0:clp_y1, clp_x0:clp_x1]
    try:
        betters = mycv2.HoughCircles(norm_closed, mycv2.HOUGH_GRADIENT, 10, r, param1=200, param2=6*r, minRadius=r - search_extend_size, maxRadius=r + search_extend_size)[0].astype(numpy.uint16)
        better = betters[0]
    except:
        better = (circle[0] - clp_x0, circle[1] - clp_y0, circle[2])
    mask_indexes = skid.circle(better[1], better[0], better[2] - remove_rim, shape=clipped.shape)
    clipped_bw = crop(norm_closed, mask_indexes)
    clipped_bw = mycv2.equalizeHist(clipped_bw)
    clip_color = crop(clip_color, mask_indexes)
    cropped = crop(clipped, mask_indexes, False)
    return clipped_bw, clip_color, cropped


def findROIs(bw, color):
    min_r = min(bw.shape[:2]) / 4
    circles = mycv2.HoughCircles(bw, mycv2.HOUGH_GRADIENT, 4, min_r, param1=200, param2=min_r, minRadius=min_r, maxRadius=2 * min_r)[0]
    circles = circles.astype(numpy.uint16)
    best_r = circles[0][2]
    Helper.logText("initial number of ROIs %d" % circles.shape[0])
    filtered_circles = circles[circles[:,2] > 0.8 * best_r]
    Helper.logText("number rough ROIs %d" % filtered_circles.shape[0])
    circles_left2right = sorted(filtered_circles, key=lambda c:c[0])
    better = [refine_circle_ROI(bw, color, c) for c in circles_left2right]
    return better


def find_colonies1(roi, roi_color):
    _, threshold = mycv2.threshold(roi, 200, 255, mycv2.THRESH_BINARY)
    morphology = mycv2.morphologyEx(threshold, mycv2.MORPH_OPEN, numpy.ones((3, 3), dtype=int))
    border = morphology - mycv2.erode(morphology, None)
    dt = mycv2.distanceTransform(border, 2, 3).astype(numpy.uint8)
    dt = mycv2.equalizeHist(dt)
    Helper.log('distanceTransform', dt)
    __, dt_trash = mycv2.threshold(dt, 100, 255, mycv2.THRESH_BINARY)
    border = mycv2.dilate(dt_trash, None, iterations=5) - dt_trash
    # dt_trash = dt_trash.astype(numpy.int32)
    # cols = mycv2.watershed(roi_color, dt_trash)
    # cols[cols < 0] = 0
    # cols = cols.astype(numpy.uint8)
    cols = border
    return cols


def find_colonies2(roi, roi_color, min_rad=3, max_rad=15):
#    blurred = mycv2.blur(roi, (10,10))
#    thr_clp = mycv2.Canny(blurred, 20, 100)
    circles = mycv2.HoughCircles(roi, mycv2.HOUGH_GRADIENT, 4, min_rad, param1=250, param2=45, minRadius=min_rad, maxRadius=max_rad)[0]
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


def find_colonies3(img):
    # Setup SimpleBlobDetector parameters.
    params = mycv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 150
    params.maxThreshold = 255
    # Filter by Area.
    params.filterByArea = False
    params.minArea = 10
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    #params.filterByInertia = True
    #params.minInertiaRatio = 0.01
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


OUTPUT_DIR = r"C:\code\colony-profile\output\cfu4good"
#PATH_NAME = r"V:\2\Camera\5\IMG_20151125_182927.jpg" # single plate (on stand)
#PATH_NAME = r"V:\2\Camera\5\IMG_20151125_183446.jpg"  # two plates (left one cut)
#PATH_NAME = r"c:\Users\refael\Downloads\2015-11-29.jpg"  # small cut plate
# PATH_NAME = r"V:\2\Camera\1\IMG_20151104_165813.jpg"  # tiny plate
#PATH_NAME = r"V:\2\Camera\2\IMG_20151104_171622.jpg"  # tiny plate
#PATH_NAME = r"C:\code\colony-profile\c4\IMG_2670.JPG"  # Lizi
PATH_NAME = r"C:\code\colony-profile\c4\IMG_2942.JPG"  # Christina
#PATH_NAME = r"V:\2\Camera\2\IMG_20151104_171616.jpg"  # small plate
#PATH_NAME = r"V:\2\Camera\4\IMG_20151105_142456_data.jpg"

# IMGs = [
# r"V:\2\Camera\5\IMG_20151125_182156.jpg",
# r"V:\2\Camera\5\IMG_20151125_182207.jpg",
# r"V:\2\Camera\5\IMG_20151125_182218.jpg",
# r"V:\2\Camera\5\IMG_20151125_182228.jpg",
# ]


def blowup(img):
    pre = img
    split = mycv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    b_split = [clahe.apply(c) for c in split]
    # b16 = numpy.multiply(b_split, b_split, dtype=numpy.uint16)
    # b8 = (b16 >> 8).astype(numpy.uint8)
    # b_split = [clahe.apply(c) for c in b8]
    img = cv2.merge(b_split)
    Helper.log('blowup', [pre, img])
    l = mycv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)[:,:,1]
    l = deluminate(l)
    th, mask = cv2.threshold(l, 128, 255, cv2.THRESH_BINARY)
    l2 = cv2.multiply(l, mask/200)
    Helper.log('blowup', [l, mask, l2, cv2.subtract(l, l2)])
    return l2, img


orig = mycv2.imread(PATH_NAME)
if orig.shape[0] < orig.shape[1]:
    orig = mycv2.transpose(orig)
blown_bw, blown_color = blowup(orig)



exit(0)

rois = findROIs(blown_bw, blown_color)
roi_bw, roi_color, cropped = rois[0]
Helper.log('found ROI', roi_color)

find_colonies3(roi_bw)

colonies2 = find_colonies2(roi_bw, roi_color)
chans = cv2.split(roi_color)
colonies2 = mycv2.merge([chans[0], chans[1], mycv2.add(chans[2], colonies2)])
Helper.log('colonies 2', colonies2)

colonies1 = find_colonies1(roi_bw, roi_color)
Helper.log('colonies 1', colonies1)
colonies2 = mycv2.merge([mycv2.add(chans[0], colonies1), chans[1], chans[2]])
Helper.log('colonies 1', colonies1)

# max10 = (pow(2, 10) - 1)
# max16 = (pow(2, 16) - 1)
# lbl = lbl * (max16 / ncc)
# lbl <<= 5
#
# lbl = lbl.astype(numpy.int32)
# lbl[border == 255] = max16
#
# lbl[lbl == -1] = 0
# result = max16 - lbl
#col_raster1 = mycv2.add(roi_color[:,:,2], colonies / 4)
#col_raster2 = mycv2.add(roi_color[:,:,1], colonies1 / 4)
#bgr_chans = mycv2.split(roi_color)
#output_stack = mycv2.merge(bgr_chans + [col_raster1, col_raster2])
#output_stack = mycv2.merge([bgr_chans[0], col_raster1, col_raster2])
#Helper.log('segment_on_dt', output_stack)
#myShow('segment_on_dt', output_stack)

# filename = os.path.basename(PATH_NAME)
# filename_no_ext, ext = os.path.splitext(filename)
# new_filename = filename + mycv2.time_stamp + ".tiff"
# out_filename = os.path.join(OUTPUT_DIR, new_filename)
# mycv2.imwrite(out_filename, output_stack)
