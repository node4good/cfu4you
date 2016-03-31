from cfu4you_lib import *

from datetime import datetime
import xlsxwriter
import math
import io
import cv2
import numpy
import os
import pandas

MAX_HEIGHT = 40.0
MAX_WIDTH = 10.0
ADJUSTED_HEIGHT = (MAX_HEIGHT - 1) * 1.4 - 2.0
ADJUSTED_WIDTH = (MAX_WIDTH - 1) * 8.0 - 2.0
DEFECT_MIN_SIZE = 2
RECT_SUB_PIX_PADDING = 20
K_RECT_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CIRCLE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
ADAPTIVE_BLOCK_SIZE = 51
ALL_CONTOURS = -1



def find_colonies1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_f = mycv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, 0)
    threshold_p = threshold_f.astype(numpy.uint8)
    threshold = cv2.erode(threshold_p, K_RECT_3)
    Helper.log_overlay(img, threshold)
    r1, K1 = get_normal_kernel(img, threshold, 0.2)

    morphed = mycv2.morphologyEx(threshold, cv2.MORPH_OPEN, K1)
    Helper.log_overlay(img, morphed)
    bg_mask = mycv2.dilate(morphed, K_CIRCLE_7)
    Helper.log_overlay(img, bg_mask)
    bg_maskC3 = cv2.merge([bg_mask, bg_mask, bg_mask])
    masked_img = cv2.bitwise_and(img, bg_maskC3)
    masked_img_eq = equalizeMulti(img)
    Helper.log_pic(masked_img_eq)

    dist_transform = mycv2.distanceTransform(threshold, cv2.DIST_L2, cv2.DIST_MASK_5)
    _, best_fg = mycv2.threshold(dist_transform, r1, 255, cv2.THRESH_BINARY)
    best_fg = best_fg.astype(numpy.uint8)
    Helper.log_overlay(img, best_fg)

    best_fg_stats = ContourStats.find_contours(best_fg, lambda r: (4 < r.area))
    markers = numpy.zeros(gray.shape, dtype=numpy.int32)
    for i, s in enumerate(best_fg_stats, 2):
        cv2.drawContours(markers, [s.contour], ALL_CONTOURS, i, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    masked_markers = mycv2.multiply(markers, bg_mask/255, dtype=cv2.CV_32S)
    masked_markers[gray == 0] = 1
    Helper.log_overlay(img, masked_markers)
    markers_after_watershed = mycv2.watershed(masked_img_eq, masked_markers)
    Helper.log_pics([markers_after_watershed])
    markers_culled = delete_bad_labels(markers_after_watershed)
    Helper.log_pics([markers_culled])
    markers_bin = numpy.zeros(gray.shape, dtype=numpy.uint8)
    markers_bin[markers_culled > 2] = 255
    Helper.log_overlay(img, markers_bin)
    r, K2 = get_normal_kernel(img, markers_bin)
    markers_bin2 = cv2.morphologyEx(markers_bin, cv2.MORPH_OPEN, K2)
    Helper.log_overlay(img, markers_bin2)
    Helper.log('markers diff', [cv2.absdiff(markers_bin, markers_bin2)])

    stats = ContourStats.find_contours(markers_bin2, lambda r: 16 < r.area < 3000)
    stats.sort(key=lambda r: r.area, reverse=True)
    outlines = numpy.zeros(img.shape, numpy.int8)
    for i, s in enumerate(stats):
        i_ = 20 * (i % 8) + 90
        cv2.drawContours(outlines, [s.contour], ALL_CONTOURS, (i_, -i_, 255), thickness=1, lineType=cv2.LINE_8)
    Helper.log('color3', [cv2.add(img, outlines, dtype=cv2.CV_8UC3)])
    return stats


def delete_bad_labels(markers_after_watershed):
    markers1 = numpy.add(markers_after_watershed, 1).astype(numpy.uint16)
    histogram = numpy.bincount(markers1.flat)
    bad_labels_bool = enumerate(histogram > 10000)
    bad_labels_pair = filter(lambda x: x[1], bad_labels_bool)
    bad_labels = map(lambda x: x[0], bad_labels_pair)
    Helper.logText('bad_labels {0}', repr(bad_labels))
    markers_culled = markers_after_watershed.copy()
    markers_culled.flat[numpy.in1d(markers1, bad_labels)] = 0
    return markers_culled


def get_normal_kernel(img, mask, b=1/3.0):
    stats = ContourStats.find_contours(mask, lambda r: (2 ** 3 < r.area < 2 ** 12 and r.roundness < 0.2))
    markers = numpy.zeros(mask.shape, dtype=numpy.uint8)
    for i, s in enumerate(stats, 1):
        cv2.drawContours(markers, [s.contour], -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_4)
    Helper.log_overlay(img, markers)
    rads = map(lambda s: s.radius, stats)
    avg = numpy.average(rads)
    std = numpy.std(rads)
    lim = avg * b
    Helper.logText('radius limit: sigma({2:.2f}), avg({0:.2f}) * {1:.2f}  = {3}', avg, b, std, lim)
    k_size = int(lim) * 2 + 1
    K = mycv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return lim, K


def churn(roi, stats):
    """

    :param roi:
    :type roi:
    :param stats:
    :type stats: List[ContourStats]
    :return:
    :rtype:
    """
    img_eq = roi
    roi_marked = img_eq.copy()

    writer = pandas.ExcelWriter(Helper.OUTPUT_PREFIX + '.xlsx', engine='xlsxwriter')
    workbook = writer.book
    workbook.default_format_properties['valign'] = 'top'
    worksheet = workbook.add_worksheet()
    worksheet.set_zoom(200)
    xfmt = workbook.add_format()
    xfmt_red = workbook.add_format()
    xfmt_red.set_font_color('red')
    xfmt_magenta = workbook.add_format()
    xfmt_magenta.set_font_color('magenta')
    xfmt_bold = workbook.add_format()
    xfmt_bold.set_font_color('green')
    xfmt_bold.set_bold(True)
    worksheet.set_column(7, 7, MAX_WIDTH)
    Helper.logText("prepare excel")

    cp = roi.copy()
    cv2.drawContours(cp, map(lambda s: s.contour, stats), -1, (255, 0, 0), thickness=1, lineType=cv2.LINE_8)

    red2 = []
    green2 = []
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
        one_side_padding = RECT_SUB_PIX_PADDING / 2
        offset = (s.offset[0] + one_side_padding, s.offset[1] + one_side_padding)
        cv2.drawContours(slc, [s.contour], -1, (255, 0, 0), thickness=1, offset=offset)
        image_arg1 = make_image_arg(s, slc)
        worksheet.insert_image(i, 7, str(i), image_arg1)
        fmt = xfmt if s.is_size else xfmt_magenta
        fmt = fmt if s.is_round else xfmt_red
        worksheet.set_row(i, MAX_HEIGHT, fmt)
        worksheet.write_number(i, 0, i)
        worksheet.write_number(i, 1, s.cx)
        worksheet.write_number(i, 2, s.cy)
        worksheet.write_number(i, 3, s.width)
        worksheet.write_number(i, 4, s.height)
        worksheet.write_number(i, 5, s.area)
        worksheet.write_number(i, 6, s.perimeter)
        # 7 - Marked
        worksheet.write_number(i, 8, s.count, xfmt_bold)
        worksheet.write_number(i, 9, s.reduced_defects)
        worksheet.write_number(i, 10, s.roundness)
        worksheet.write_boolean(i, 11, s.is_size)

    worksheet.add_table(0, 0, len(stats)+1, 11, {
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
            {'header': 'pic marked'},
            {'header': 'COUNT', 'total_function': 'sum'},
            {'header': 'defects No.'},
            {'header': 'roundness'},
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

    data = [s.__getstate__() for s in stats]
    df = pandas.DataFrame(data)
    df.to_excel(writer, index=False, sheet_name='raw')
    writer.sheets['raw'].add_table(0, 0, len(stats) + 1, df.columns.size, {'banded_rows': False})
    Helper.logText(u"added raw data")

    writer.save()
    Helper.logText(u"saved excel")

    # markers_ws = mycv2.watershed(roi_color, markers)
    Helper.logText(u"churn: raw {0:d}\t\tfinal {1:d}".format(len(stats), len(green2)))
    # cols = cv2.watershed(roi_color, dt_trash)

    return roi_marked


def make_image_arg(s, slc):
    __, buf1 = cv2.imencode('.jpg', slc)
    image_data1 = io.BytesIO(buf1)
    image_arg1 = {'image_data': image_data1, 'positioning': 1}
    a_height = s.height + RECT_SUB_PIX_PADDING
    a_width = s.width + RECT_SUB_PIX_PADDING
    if ADJUSTED_HEIGHT < a_height or ADJUSTED_WIDTH < a_width:
        fact_h = (ADJUSTED_HEIGHT) / (a_height + 16)
        fact_w = (ADJUSTED_WIDTH) / (a_width + 4)
        image_arg1['x_scale'] = image_arg1['y_scale'] = min(fact_h, fact_w)
    return image_arg1


def process_file(filename):
    # noinspection PyUnresolvedReferences
    ts = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    Helper.logText("*************** {0:s} - {1} ***************".format(filename, ts))
    file_path = os.path.join(DIR_NAME, filename)
    just_name, _ext = os.path.splitext(filename)
    Helper.OUTPUT_PREFIX = os.path.join(OUTPUT_DIR, '{0}_{1}'.format(ts, just_name))
    orig = mycv2.imread(file_path)
    if orig.shape[0] < orig.shape[1]:
        orig = numpy.rot90(orig, 3)
        Helper.logText("rotated")
    masked, blown = blowup(orig)
    Helper.log_pics([masked])
    Helper.log_pics([blown])
    rois = findROIs(masked)
    roi = rois[0]
    [roi_color] = cropROI(roi, blown)
    # roi_color = blown
    Helper.log_pics([roi_color])
    st = find_colonies1(roi_color)
    if len(st) == 0: return
    # data = [s.__getstate__() for s in st]
    # df = pandas.DataFrame(data)
    # df.to_csv(filename + '.csv')
    colonies1_merged = churn(roi_color, st)
    Helper.log(file_path, colonies1_merged)



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
# DIR_NAME = r"C:\code\6broad\.data\CFU\RB\images_for_Refael"
# DIR_NAME = r"C:\code\6broad\.data\CFU\nj"
DIR_NAME = r"C:\code\6broad\.data\CFU\CFU(2)"
# DIR_NAME = r"C:\code\6broad\colony-profile\output\cfu4good\RB"
files = os.listdir(DIR_NAME)[1:2]
# random.shuffle(files)
# files = ['2016-03-28 17.51.21.jpg'] # EASY
# files = ['E072_d7.JPG'] # EASY
# files = ['E072_g7.JPG'] # HARD
# files = ['2016-03-21-17-46-27_E072_d7_roi_color.jpg'] # cropped

OUTPUT_DIR = r"C:\code\6broad\colony-profile\output\cfu4good\ZBA"


for f1 in files:
    process_file(f1)

