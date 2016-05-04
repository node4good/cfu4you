from cfu4you_lib import *
from datetime import datetime
import math
from io import BytesIO as StringIO
import os
import vlogging
import pandas
CUR_DIR = os.path.dirname(__file__)

js_file = os.path.join(CUR_DIR, r'..\assets\cfugui.js')
with open(js_file, 'r') as content_file:
    script_code = content_file.read()


def find_colonies1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_f = mycv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, 0)
    threshold_p = threshold_f.astype(numpy.uint8)
    threshold = cv2.erode(threshold_p, K_RECT_3)
    Helper.log_overlay(img, threshold)
    r1, K1, bp1 = get_sample_parameters(img, threshold, 0.2)
    _, bp_t = cv2.threshold(bp1, bp1[0,0], 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bp_tm = cv2.morphologyEx(bp_t, cv2.MORPH_CLOSE, K1)
    bp_tm2 = cv2.morphologyEx(bp_tm, cv2.MORPH_OPEN, K1)
    Helper.log_pics([bp1, bp_t, bp_tm, bp_tm2])
    morphed = mycv2.morphologyEx(threshold, cv2.MORPH_OPEN, K1)
    bg_mask_inv = mycv2.dilate(morphed, K_CIRCLE_21)
    bg_mask = mycv2.bitwise_not(bg_mask_inv)
    Helper.log_overlay(img, bg_mask)
    bg_mask_3c = cv2.merge([bg_mask, bg_mask, bg_mask])
    bg_masked_img = cv2.bitwise_and(img, bg_mask_3c)
    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([bg_masked_img], [i], None, [256], [0, 256])
    #     line, = plt.plot(histr, color=col)
    #     line.axes.set_yscale('log', basey=2)
    #     plt.xlim([0, 256])
    # plt.show()
    masked_img_eq = equalizeMulti(bg_masked_img)

    dist_transform = mycv2.distanceTransform(threshold, cv2.DIST_L2, cv2.DIST_MASK_5)
    _, best_fg = mycv2.threshold(dist_transform, r1, 255, cv2.THRESH_BINARY)
    best_fg = best_fg.astype(numpy.uint8)
    fg_mask_3c = cv2.merge([best_fg, best_fg, best_fg])
    fg_masked_img = cv2.bitwise_and(img, fg_mask_3c)
    # bg_hist = cv2.calcHist(bg_masked_img, [0, 1, 2], bg_mask_3c, [256], [0, 256])
    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([fg_masked_img], [i], None, [256], [0, 256])
    #     line, = plt.plot(histr, color=col)
    #     line.axes.set_yscale('log', basey=2)
    #     plt.xlim([0, 256])
    # plt.show()
    Helper.log_overlay(img, best_fg)
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
    # mask.size() == imsize & & mask.channels() == 1
    h_m = cv2.bitwise_and(h, best_fg)
    bg_hist = cv2.calcHist([h_m], [0], best_fg, [256], [0, 256])
    bg_hist_norm = cv2.normalize(bg_hist, bg_hist, 0, 255, cv2.NORM_MINMAX)
    bp = cv2.calcBackProject([h], [0], bg_hist_norm, [0, 256], 1)
    Helper.log_pic(bp)

    best_fg_stats = ContourStats.find_contours(bp_tm, lambda r: (4 < r.area))
    markers = numpy.zeros(gray.shape, dtype=numpy.int32)
    for i, s in enumerate(best_fg_stats, 2):
        cv2.drawContours(markers, [s.contour], ALL_CONTOURS, i, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    masked_markers = mycv2.multiply(markers, bg_mask_inv/255, dtype=cv2.CV_32S)
    masked_markers[gray == 0] = 1
    markers_after_watershed = mycv2.watershed(masked_img_eq, masked_markers)
    Helper.log_pics([markers_after_watershed])
    markers_culled = delete_bad_labels(markers_after_watershed)
    Helper.log_pics([markers_culled])
    markers_bin = numpy.zeros(gray.shape, dtype=numpy.uint8)
    markers_bin[markers_culled > 2] = 255
    r, K2, bp2 = get_sample_parameters(img, markers_bin)
    markers_bin2 = cv2.morphologyEx(markers_bin, cv2.MORPH_OPEN, K2)

    stats = ContourStats.find_contours(markers_bin2, lambda r: 16 < r.area < 3000)
    stats.sort(key=lambda r: r.area, reverse=True)
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


def get_sample_parameters(img, mask, b=1 / 2.5):
    stats = ContourStats.find_contours(mask)
    stats_normal = filter(lambda r: (2 ** 3 < r.area < 2 ** 12 and r.roundness < 0.2), stats)
    draw_stats(mask, stats_normal)
    rads = map(lambda s: s.radius, stats_normal)
    avg = numpy.average(rads)
    rk_avg = int(avg) * 2 - 1
    K_AVG = mycv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk_avg, rk_avg))
    std = numpy.std(rads)
    lim = avg * b
    Helper.logText('radius limit: sigma({2:.2f}), avg({0:.2f}) * {1:.2f}  = {3}', avg, b, std, lim)
    k_size = int(math.ceil(lim)) * 2 + 1
    K = mycv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    stats_normal.sort(key=lambda s: s.area)
    mid_point = len(stats_normal) / 2
    mid = map(lambda s: s.contour, stats_normal[mid_point-5:mid_point+5])
    hist_mask = numpy.zeros(mask.shape, numpy.uint8)
    cv2.drawContours(hist_mask, mid, -1, 255, thickness=cv2.FILLED)
    hist_mask_d = cv2.dilate(hist_mask, K_RECT_3)
    hist_mask_e = cv2.erode(hist_mask_d, K_CIRCLE_7)
    borders = hist_mask_d - hist_mask_e
    Helper.log_overlay(img, borders)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_AVG)
    Helper.log_overlay(img, mask_morph)

    h_bins = 180
    s_bins = 255
    histSize = [s_bins]

    h_range = [0, 179]
    s_range = [0, 255]
    ranges = s_range

    channels = [1]

    colony_hist = cv2.calcHist([hsv], channels, borders, histSize, ranges)
    cv2.normalize(colony_hist, colony_hist, 0, 255, cv2.NORM_MINMAX)
    bp = cv2.calcBackProject([hsv], channels, colony_hist, ranges, 1)
    return lim, K, bp


def draw_stats(img, *stats):
    masks = [img]
    for stat in stats:
        mask = numpy.zeros(list(img.shape[0:2]), dtype=numpy.uint8)
        for i, s in enumerate(stat, 1):
            cv2.drawContours(mask, [s.contour], -1, 128 + (i % 16) * 8, thickness=cv2.FILLED, lineType=cv2.LINE_4)
        masks.append(mask)
    Helper.log_pics(masks)


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
    image_data_mark = StringIO(buf_mark)
    worksheet2 = workbook.add_worksheet()
    worksheet2.insert_image(0, 0, 'marked', {'image_data': image_data_mark, 'positioning': 2})
    worksheet2.set_zoom(50)
    Helper.logText("encoded marked")

    _, buf_raw = cv2.imencode('.jpg', roi)
    image_data_raw = StringIO(buf_raw)
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
    image_data1 = StringIO(buf1)
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
    masked = blowup_threshold(orig)
    Helper.log_pics([masked])
    rois = findROIs(masked)
    roi = rois[0]
    [roi_color_pre] = cropROI(roi, orig)
    roi_color = blowup_roi(roi_color_pre)
    Helper.log("roi_color", roi_color)
    st = find_colonies1(roi_color)
    if len(st) == 0: return
    # colonies1_merged = churn(roi_color, st)
    # Helper.log(file_path, colonies1_merged)
    # data = [s.__getstate__() for s in st]
    # df = pandas.DataFrame(data)
    file_name = os.path.join(OUTPUT_DIR, "output" + Helper.time_stamp + ".html")
    img_file = file_name + ".png"
    js_data = json.dumps(st, cls=NumpyAwareJSONEncoder)
    js_string = """
<!DOCTYPE html>
<html>

<head>
<script src="http://cdnjs.cloudflare.com/ajax/libs/fabric.js/1.6.1/fabric.min.js"></script>
</head>

<body>
<canvas id="c"></canvas>
<script>
var d = {0};
var img_src = "{1}";
</script>
<script src="file://{4}"></script>
<a href="file://{3}">log</a></br>
</body>

</html>
""".format(js_data, img_file.replace('\\', '/'), script_code, Helper.htmlfile, js_file.replace('\\', '/'))
    cv2.imwrite(img_file, roi_color)
    with open(file_name, 'w') as output_file:
        output_file.write(js_string)
    webbrowser.open_new_tab(file_name)



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
# DIR_NAME = r"C:\code\6broad\.data\CFU\nj"
# DIR_NAME = r"C:\Users\refael\Downloads"
# DIR_NAME = r"C:\code\6broad\colony-profile\output\cfu4good\RB"
# files = os.listdir(DIR_NAME)[1:2]
# random.shuffle(files)
# files = ['2016-03-28 17.51.21.jpg'] # EASY
files = ['E072_d7.JPG'] # EASY
# files = ['E072_g7.JPG'] # HARD
# files = ['DSCF0010_rif_inh_1-10.JPG'] # HARD
# files = ['2016-03-21-17-46-27_E072_d7_roi_color.jpg'] # cropped

OUTPUT_DIR = r"C:\code\6broad\colony-profile\output\cfu4good\ZBA"


for f1 in files:
    process_file(f1)

