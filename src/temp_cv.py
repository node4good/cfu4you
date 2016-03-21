from cfu4you_lib import *

from datetime import datetime
import xlsxwriter
import math
import io
import cv2
import numpy
import os


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


def find_colonies1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold = mycv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADPT_BLOCK_SIZE, 0).astype(numpy.uint8)
    eroded = mycv2.erode(threshold, K_CROSS_3, iterations=3)
    morhped = mycv2.morphologyEx(eroded, cv2.MORPH_OPEN, K_CROSS_5, iterations=2)
    mrg = cv2.merge([threshold, eroded, morhped])[0:900, 0:900, :]
    Helper.log("threshold, eroded, morhped", mrg)
    sure_bg = mycv2.dilate(morhped, K_CIRC_5, iterations=3)
    Helper.log_overlay(img, sure_bg)
    sure_bg_inv = mycv2.bitwise_not(eroded)
    Helper.log('sure_bg_inv', [sure_bg_inv])
    bg_inv_dilated = mycv2.dilate(sure_bg_inv, K_CIRC_3, iterations=2)
    Helper.log('bg_inv_dilated', [bg_inv_dilated])
    dist_transform = mycv2.distanceTransform(morhped, cv2.DIST_L2, 5)
    _, sure_fg = mycv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(numpy.uint8)
    Helper.log("sure_fg, bg_inv_dilated, eroded", cv2.merge([sure_fg, bg_inv_dilated, eroded]))
    stats = ContourStats.find_contours(sure_fg, lambda r: (4 < r.area))
    markers = numpy.zeros(gray.shape, dtype=numpy.int32)
    color3 = img.copy()
    for i, s in enumerate(stats, 1):
        cv2.drawContours(markers, [s.contour], -1, i, thickness=cv2.FILLED, lineType=cv2.LINE_8)
        i_ = 10 * (i % 16) + 90
        cv2.drawContours(color3, [s.contour], -1, (0,0,i_), thickness=1, lineType=cv2.LINE_8)
    bg_stats = ContourStats.find_contours(sure_bg_inv, lambda r: (500 < r.area))
    for i, s in enumerate(bg_stats, i + 1):
        cv2.drawContours(markers, [s.contour], -1, i, thickness=1, lineType=cv2.LINE_8)
        i_ = 10 * (i % 16) + 90
        cv2.drawContours(color3, [s.contour], -1, (0,i_,0), thickness=1, lineType=cv2.LINE_8)
    Helper.log('markers', [markers])
    Helper.log('color3 1', [color3])

    mask_3c = mycv2.merge([sure_bg_inv, sure_bg_inv, sure_bg_inv])
    masked = mycv2.subtract(img, mask_3c)
    Helper.log('mask', [masked])
    markers = mycv2.watershed(masked, markers)
    contours2 = numpy.zeros(gray.shape, dtype=numpy.uint8)
    contours2[markers == -1] = 255
    contours3 = numpy.ones(gray.shape, dtype=numpy.uint8) * 255
    contours3[markers == -1] = 0
    Helper.log('contours3', [contours3])
    retval, labels, stats, centroids = mycv2.connectedComponentsWithStats(contours3, connectivity=4)
    labels2 = labels.copy()
    v = enumerate(stats)
    v = filter(lambda x: x[1][4] > 20000, v)
    vi = map(lambda x: x[0], v)
    labels2.flat[numpy.in1d(labels2, vi)] = 0
    Helper.logText('connected count {0} valid {1}', len(stats), len(vi))
    Helper.log('labels2', [labels2.astype(numpy.uint8)])
    contours2_d = mycv2.dilate(contours2, K_CIRC_3, iterations=1)
    labels2[contours2_d == 255] = 0
    labels3 = labels2.astype(numpy.uint8)
    labels3 = mycv2.erode(labels3, K_CIRC_3, iterations=4)
    labels3 = mycv2.dilate(labels3, K_CIRC_3, iterations=4)
    Helper.log('labels3', [labels3])
    stats = ContourStats.find_contours(labels3, lambda r: (16 < r.area < 10000))
    stats.sort(key=lambda r: r.area, reverse=1)
    contours2 = numpy.zeros(img.shape, dtype=numpy.uint8)
    for i, s in enumerate(stats):
        i_ = 7 * ((i % 64)) + 63
        cv2.drawContours(contours2, [s.contour], -1, (0,255,i_), thickness=2, lineType=cv2.LINE_8)
        cv2.putText(contours2, str(i), (s.cx, s.cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,i_), thickness=2, lineType=cv2.LINE_8)
    Helper.log('contours', [contours2])
    return stats


def churn(roi, stats):
    img_eq = equalizeMulti(roi)
    roi_marked = img_eq.copy()

    workbook = xlsxwriter.Workbook(Helper.OUTPUT_PREFIX + '.xlsx')
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
        worksheet.write_number(i, 11, s.roundness2)
        worksheet.write_number(i, 12, s.regularity)
        worksheet.write_number(i, 13, s.reduced_defects)
        worksheet.write_boolean(i, 14, s.is_round)
        worksheet.write_boolean(i, 15, s.is_size)

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
            {'header': 'roundness2'},
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
    Helper.logText(u"saved excel")

    # markers_ws = mycv2.watershed(roi_color, markers)
    Helper.logText(u"churn: raw {0:d}\t\tfinal {1:d}".format(len(stats), len(green2)))
    # cols = cv2.watershed(roi_color, dt_trash)

    return roi_marked


def process_file(filename):
    ts = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    Helper.logText("*************** {0:s} - {1} ***************".format(filename, ts))
    file_path = os.path.join(DIR_NAME, filename)
    just_name, _ext = os.path.splitext(filename)
    Helper.OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, '{0}_{1}'.format(ts, just_name))
    orig = mycv2.imread(file_path)
    if orig.shape[0] < orig.shape[1]:
        orig = numpy.rot90(orig, 3)
        Helper.logText("rotated")
    masked = blowup(orig)
    Helper.log_pics([masked])
    rois = findROIs(masked)
    roi = rois[0]
    [roi_color] = cropROI(roi, orig)
    Helper.log_pics([roi_color])
    st = find_colonies1(roi_color)
    # data = [s.__getstate__() for s in st]
    # df = pandas.DataFrame(data)
    # df.to_csv(filename + '.csv')
    # colonies1_merged = churn(roi_color, st)
    # Helper.log(file_path, colonies1_merged)




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

