from cfu4you_lib import *

from datetime import datetime
import xlsxwriter
import math
import io
import cv2
import numpy
import os
import pandas

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
K_CIRC_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
K_CIRC_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
K_CIRC_11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
K_CIRC_13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
K_CIRC_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
K_CIRC_25 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
K_SQ_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CROSS_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
K_CROSS_5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
ADPT_BLOCK_SIZE = 51
ALL_CNTS = -1



def segment_transform(mask, pred=None):
    stats = ContourStats.find_contours(mask, pred)
    markers = numpy.zeros(mask.shape, dtype=numpy.uint8)
    for i, s in enumerate(stats, 1):
        cv2.drawContours(markers, [s.contour], -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_4)
    Helper.log_pics([markers])
    return markers



def segment_transform2(mask, pred=None):
    _, labels, stats, centroids = mycv2.connectedComponentsWithStats(mask, connectivity=4)
    v = enumerate(stats)
    v = filter(lambda x: pred(x[1][4]), v)
    vi = map(lambda x: x[0], v)
    out = numpy.zeros(mask.shape, dtype=numpy.uint8)
    out.flat[numpy.in1d(labels, vi)] = 255
    Helper.logText('connected count {0} valid {1}', len(stats), len(vi))
    Helper.log_pics([out])
    return out


def segment3(color, markers):
    markers = mycv2.watershed(color, markers)
    mask = markers.copy()
    mask[mask == -1] = 0
    Helper.log_pics([mask])
    _, labels, stats, centroids = mycv2.connectedComponentsWithStats(mask.astype(numpy.uint8), connectivity=4)
    labels = labels.astype(numpy.uint16)
    Helper.log_pics([labels + 20000])
    good = []
    for i in range(1, len(stats)):
        s = stats[i]
        area_circularity = (s[2] * s[3] * math.pi) / (s[4] * 4.0)
        r = abs(math.log(area_circularity, 1.1))
        if s[4] > 4000: continue
        good.append(i)
    out = []
    for i in good:
        label = numpy.zeros(labels.shape, dtype=numpy.uint8)
        label[labels == i] = 255
        ret = cv2.findContours(label, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        img_marked, contours, [hierarchy] = ret
        cnt = ContourStats.from_contour((contours[0], hierarchy[0]), i)
        if cnt is None: continue
        out.append(cnt)
    Helper.logText('segment3 connected count {0}', len(out))
    return labels, out


def find_colonies1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold_f = mycv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADPT_BLOCK_SIZE, 0)
    threshold = threshold_f.astype(numpy.uint8)
    eroded = cv2.erode(threshold, K_CIRC_11)
    dilated = cv2.dilate(eroded, K_CIRC_15)
    Helper.log_overlay(img, dilated)
    dist_transform = cv2.distanceTransform(threshold, cv2.DIST_L2, 5)
    _, best_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    best_fg = best_fg.astype(numpy.uint8)
    Helper.log_overlay(img, best_fg)
    mask = dilated
    mask_c3 = cv2.merge([mask, mask, mask])
    Helper.log_overlay(img, mask)
    masked_img = cv2.bitwise_and(img, mask_c3)
    masked_img_eq = equalizeMulti(masked_img)

    best_fg_stats = ContourStats.find_contours(best_fg, lambda r: (4 < r.area))
    markers = numpy.zeros(gray.shape, dtype=numpy.int32)
    for i, s in enumerate(best_fg_stats, 2):
        cv2.drawContours(markers, [s.contour], ALL_CNTS, i, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    masked_markers = mycv2.multiply(markers, mask/255, dtype=cv2.CV_32S)
    Helper.log_overlay(img, masked_markers)
    masked_markers[1,1] = 1 # add a marker for the BG

    # labels, stats = segment3(masked_img, masked_markers)
    markers_after_watershed = mycv2.watershed(masked_img_eq, masked_markers)
    Helper.log_pics([markers_after_watershed])
    markers_bin = numpy.zeros(gray.shape, dtype=numpy.uint8)
    markers_bin[markers_after_watershed < 2] = 255
    Helper.log_overlay(img, markers_bin)
    markers_bin2 = cv2.morphologyEx(markers_bin, cv2.MORPH_OPEN, K_CIRC_11)
    Helper.log_overlay(img, markers_bin2)
    Helper.log('markers diff', [cv2.absdiff(markers_bin, markers_bin2)])

    stats = ContourStats.find_contours(markers_bin2, lambda r: 8 < r.area < 3000)
    stats.sort(key=lambda r: r.area, reverse=True)
    outlines = numpy.zeros(img.shape, numpy.int8)
    for i, s in enumerate(stats):
        i_ = choose_color(i)
        cv2.drawContours(outlines, [s.contour], ALL_CNTS, (i_,-50,i_), thickness=1, lineType=cv2.LINE_8)
    Helper.log('color3', [cv2.add(img, outlines, dtype=cv2.CV_8UC3)])
    return stats


def choose_color(i):
    return 10 * (i % 16) + 90


def churn(roi, stats):
    img_eq = roi
    roi_marked = img_eq.copy()

    # By setting the 'engine' in the ExcelWriter constructor.
    writer = pandas.ExcelWriter(Helper.OUTPUT_PREFIX + '.xlsx', engine='xlsxwriter')
    workbook = writer.book
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
    worksheet.set_column(7, 7, MAX_WIDTH)
    Helper.logText("prepare excel")

    cp = roi.copy()
    cv2.drawContours(cp, map(lambda s:s.contour, stats), -1, (255, 0, 0), thickness=1, lineType=cv2.LINE_8)

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
        one_side_padding = RECT_SUB_PIX_PADDING / 2
        offset = (s.offset[0] + one_side_padding, s.offset[1] + one_side_padding)
        cv2.drawContours(slc, [s.contour], -1, (255, 0, 0), thickness=1, offset=offset)
        image_arg1 = make_image_arg(s, slc)
        worksheet.insert_image(i, 7, str(i), image_arg1)
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
        # 7 - Marked
        worksheet.write_number(i, 8, s.count, xfmt_bold)
        worksheet.write_number(i, 9, s.reduced_defects)
        worksheet.write_boolean(i, 10, s.is_round)
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

    data = [s.__getstate__() for s in stats]
    df = pandas.DataFrame(data)
    # df.to_excel(writer, index=False, sheet_name='raw')
    # writer.sheets['raw'].add_table(0, 0, len(stats) + 1, df.columns.size, {'banded_rows': False})
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
    Helper.log_pics([roi_color])
    st = find_colonies1(roi_color)
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
DIR_NAME = r"C:\code\6broad\.data\CFU\RB\images_for_Refael"
# DIR_NAME = r"C:\code\6broad\colony-profile\output\cfu4good\RB"
# files = os.listdir(DIR_NAME)
# random.shuffle(files)
# files = ['E072_g7.JPG'] # HARD
files = ['E072_d7.JPG'] # EASY
# files = ['2016-03-21-17-46-27_E072_d7_roi_color.jpg'] # cropped

OUTPUT_DIR = r"C:\code\6broad\colony-profile\output\cfu4good\RB"


for f1 in files:
    process_file(f1)

