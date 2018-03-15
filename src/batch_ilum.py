import os
from collections import namedtuple
from datetime import datetime
import glob
import re

import numpy as np
import pydash
# from skimage import img_as_float, img_as_ubyte, img_as_uint
# from skimage.exposure import rescale_intensity
from skimage.exposure import rescale_intensity, histogram
from skimage.filters import median
# from skimage.filters.rank import maximum, mean
from skimage.io import imread, imsave
from skimage.morphology import disk
from skimage.transform import rescale

from LogHelper import LogHelper

ROOT_DIR = r"T:\NewImaging"

NAME_PARSER = re.compile('^([^_]+)_([A-G])(\d\d)_s(\d{1,2})(?:_w(\d))?')
Frame = namedtuple('Frame', 'fullpath, filename, experiment, row, column, site, wave')


def parse_file_name(f):
    n = os.path.basename(f)
    parts = [f, n] + NAME_PARSER.split(n)[1:6]
    return Frame._make(parts)


def get_plate_files(plate):
    # example "ROOT_DIR\2-7-18 T0-48in T0-48 24in pTiGc-20\2018-02-17\59574"
    dirs = glob.glob(ROOT_DIR + "\\*\\*\\*\\")
    plate = pydash.find(dirs, lambda p: plate in p)
    files = glob.glob(plate + "\\*.tif")
    parsed = map(parse_file_name, files)
    return parsed


def calculate_stats():
    parsed = get_plate_files("59798")
    ts = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    for w in ['w2']:
        all_files = filter(lambda f: f.wave == w[1], parsed)
        accum = np.zeros((2160, 2160), dtype=np.uint32)
        # files = filter(lambda x: 's1' not in x and 's7' not in x, all_files)
        files = all_files
        nof = len(all_files)
        gmn, gmx = 2**15, 0
        for i, frame in enumerate(files, 1):
            img = imread(frame.fullpath)
            s = np.sort(img.flatten())
            LogHelper.logText(str(np.argwhere(img == img.max())))
            LogHelper.logText(','.join(map(str, s[-50:-1])))
            mn, mx = np.percentile(img, (0.1, 99.9))
            gmn = min(mn, gmn)
            gmx = max(mx, gmx)
            LogHelper.logText(
                '%3d of %d, %4d-%4d-%4d-%5d, %.0f-%.0f'
                % (i, nof, img.min(), mn, mx, img.max(), np.mean(img), np.std(img))
            )
            clip = img
            accum += clip
        LogHelper.logText('w %f,%f' % (gmn, gmx))
        filename = os.path.join(ROOT_DIR, w + '-' + ts + '-%s.tiff')
        LogHelper.logText(filename)
        # np.save(filename % 'accum', accum)
        avg_u = (accum / len(files)).astype(np.uint16)
        imsave(filename % 'avg_u', avg_u)
        smooth = median(avg_u, disk(20))
        imsave(filename % 'smooth', smooth)


def calculate_ilum():
    parsed = get_plate_files("59798")
    ts = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    for w in ['w2']:
        all_files = filter(lambda f: f.wave == w[1], parsed)
        accum = np.zeros((2160, 2160), dtype=np.uint32)
        # files = filter(lambda x: 's1' not in x and 's7' not in x, all_files)
        files = all_files
        nof = len(all_files)
        gmn, gmx = 2**15, 0
        for i, frame in enumerate(files, 1):
            img = imread(frame.fullpath)
            s = np.sort(img.flatten())
            LogHelper.logText(str(np.argwhere(img == img.max())))
            LogHelper.logText(','.join(map(str, s[-50:-1])))
            mn, mx = np.percentile(img, (0.1, 99.9))
            gmn = min(mn, gmn)
            gmx = max(mx, gmx)
            LogHelper.logText(
                '%3d of %d, %4d-%4d-%4d-%5d, %.0f-%.0f'
                % (i, nof, img.min(), mn, mx, img.max(), np.mean(img), np.std(img))
            )
            # clip = np.clip(img, 0, m1)
            clip = img
            accum += clip
        LogHelper.logText('w %f,%f' % (gmn, gmx))
        filename = os.path.join(ROOT_DIR, w + '-' + ts + '-%s.tiff')
        LogHelper.logText(filename)
        # np.save(filename % 'accum', accum)
        avg_u = (accum / len(files)).astype(np.uint16)
        imsave(filename % 'avg_u', avg_u)
        smooth = median(avg_u, disk(20))
        imsave(filename % 'smooth', smooth)


def dilum():
    dir_name = get_plate_files("59438")
    ilum = imread(dir_name + "\\w1-2018-02-15-14-03-49-smooth.tiff")
    all_files = glob.glob1(dir_name, '*.tif')
    files = all_files
    for filename in files[0:1]:
        ts = "{:%Y-%m-%d-%H-%M-%S}-".format(datetime.now())
        file_path = os.path.join(dir_name, filename)
        img = imread(file_path)
        img2 = (np.int32(img) - ilum)
        img3 = np.clip(img2, 0, None)
        img4 = rescale_intensity(img3, out_range=np.uint8).astype(np.uint8)
        out_name = os.path.join(dir_name, filename).replace('.tif', '-' + ts + '.tif')
        imsave(out_name.replace('.tif', '-d.tif'), img4)
        LogHelper.logText("*************** {0:s} - {1} ***************".format(out_name, ts))


# def dilum():
#     ilum_name = os.path.join(OUTPUT_DIR, 'ilum.npy')
#     ilum = img_as_uint(np.load(ilum_name))
#     all_files = glob.glob1(DIR_NAME, '*.TIF')
#     files = all_files
#     for filename in files[0:1]:
#         ts = "{:%Y-%m-%d-%H-%M-%S}-".format(datetime.now())
#         file_path = os.path.join(DIR_NAME, filename)
#         out_name = os.path.join(OUTPUT_DIR, filename).replace('.TIF', '-' + ts + '.TIF')
#         img = imread(file_path)
#         img2 = rescale_intensity(img, in_range=(30, 520))
#         img3 = img_as_float(img2)
#         dilum = img3 / (ilum / ilum.min())
#         fu = img_as_uint(dilum)
#         imsave(out_name.replace('.TIF', '-d.TIF'), fu)
#         fb = img_as_ubyte(dilum)
#         imsave(out_name, fb)
#         Helper.logText("*************** {0:s} - {1} ***************".format(out_name, ts))
#
#
def find_min():
    dir_name = get_plate_files("59438")
    ilum_name = os.path.join(dir_name, 'ilum.tiff')
    all_files = glob.glob1(dir_name, '*.TIF')
    files = all_files
    m = 500
    ts = "{:%Y-%m-%d-%H-%M-%S}-".format(datetime.now())
    for filename in files:
        file_path = os.path.join(dir_name, filename)
        out_name = os.path.join(dir_name, filename)
        img = imread(file_path)
        m1 = min(m, np.percentile(img, 1))
        if m != m1: LogHelper.logText("{0} - {1} ".format(m1, filename))
        m = m1
        LogHelper.logText("*************** {0} - {1} ***************".format(m, ts))


def stitch():
    DIM = 1080
    sDIM = 216 + 4
    DIMP = sDIM + 5
    plate_name = '59476'
    plates = glob.glob(r"T:\NewImaging\ref-data\exp1\*")
    plate = filter(lambda d: plate_name in d, plates).pop()
    files = glob.glob1(plate, "*_Outlines.png")
    frames = [parse_file_name(f) for f in files]
    superdim = (DIMP*4*6, DIMP*4*10, 3)
    superframe = np.ones(superdim, dtype=np.uint8) * 255
    for f in frames:
        LogHelper.logText(f.filename)
        s = int(f.site) - 1
        c = int(f.column) - 2
        r = (ord(f.row) - ord('B'))
        y = sDIM*(s % 4) + DIMP*4*c
        x = sDIM*(s / 4) + DIMP*4*r
        img = imread(plate + '\\' + f.filename)
        img = rescale(img, 0.2, multichannel=True, preserve_range=True)
        imgp = np.pad(img, ((0,4), (0,4), (0,0)), 'constant', constant_values=128)
        LogHelper.logText('{0} read to go {1}x{2}'.format(f.filename, x, y))

        superframe[x:x+imgp.shape[0], y:y+imgp.shape[1]] = imgp
        LogHelper.logText(f.filename + ' placed')

    tiff.imsave(plate + '\\59476super.png', superframe)


if __name__ == "__main__":
    calculate_ilum()
    # dilum()
    # find_min()
    # stitch()
