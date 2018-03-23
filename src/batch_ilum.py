import os
import random
import scipy
from collections import namedtuple
from datetime import datetime
import glob
import re
import cv2

import numpy as np
import pydash
# from skimage import img_as_float, img_as_ubyte, img_as_uint
# from skimage.exposure import rescale_intensity
from skimage.exposure import rescale_intensity, histogram
from skimage import filters, measure
# from skimage.filters.rank import maximum, mean
from skimage.io import imread, imsave, imshow
from skimage.external.tifffile import imsave as tifsave
from skimage.morphology import disk, square, binary_erosion, binary_dilation, binary_closing
from skimage.transform import rescale
from LogHelper import LogHelper
LogHelper.NO_LINE_TIME = True

ROOT_DIR = r"T:\NewImaging"
NAME_PARSER = re.compile('^([^_]+)_([A-G])(\d\d)_s(\d{1,2})(?:_w(\d))?')
FrameRaw = namedtuple('Frame', 'fullpath, filename, plate, experiment, row, column, site, wave')
PathNode = namedtuple('PathNode', 'root, dirs, files')


class Frame(FrameRaw):
    def shortname(self):
        return '|'.join([self.plate, self.wave, self.row, self.column, self.site])


def get_plate_files(plate_name, cnt=-1):
    def parse_file_name(f):
        n = os.path.basename(f)
        parts = [f, n, plate_name] + NAME_PARSER.split(n)[1:6]
        if len(parts[6]) == 1:
            parts[6] = '0' + parts[6]
        frm = Frame._make(parts)
        return frm

    tree = LogHelper.time(lambda: [PathNode._make(t) for t in os.walk(ROOT_DIR + "\\data", topdown=False)])
    plate = pydash.find(tree, lambda p: p.root.endswith(plate_name))
    LogHelper.logText(plate.root)
    if plate.dirs:
        files = glob.glob(plate.root + "\\*\\*.tif")
    else:
        files = glob.glob(plate.root + "\\*.tif")
    if cnt > 0:
        files = random.sample(files, cnt)
    parsed = map(parse_file_name, files)
    return parsed



def calculate_stats():
    # d = imread(r"T:\NewImaging\w2-2018-03-16-18-47-26-smooth.tiff")
    # dmn, dmd, dmx = np.percentile(d, (0.1, 50, 99.9))
    # LogHelper.logText(
    #     '%4d-%4d-%4d-%4d-%5d, %.2f-%.2f'
    #     % (d.min(), dmn, dmd, dmx, d.max(), np.mean(d), np.std(d))
    # )
    parsed = get_plate_files("59833", 30)
    for w in ['1']:
        files = filter(lambda f: f.wave == w, parsed)
        # files = filter(lambda x: 's1' not in x and 's7' not in x, files)
        nof = len(files)
        p50s = []
        mps = []
        means = []
        s2s = []
        stds = []
        for i, frame in enumerate(files, 1):
            LogHelper.logText("%3d of %d - %s" % (i, nof, frame.shortname()))
            img = imread(frame.fullpath)
            p1, p50, p99 = np.percentile(img, (1, 50, 99))
            mp = np.argmax(np.bincount(img.flat))  # type: np.uint16
            s2 = mp - img.min()
            # r1 = measure.block_reduce(img, (30, 30), func=np.std)
            # r2 = measure.block_reduce(img, (20, 20), func=np.std)
            # l = abs(filters.laplace(img, 3))
            LogHelper.logText(
                '%4d-%4d # %4d-%4d # %4d-%d-%4d #%4d-%4d' % (img.min(), img.max(), p1, p99, p50, mp, img.mean(), s2, img.std())
                # "%08.8f %08.8f" % (r1.max() / 30, r2.max() / 20)
                ### good focus s > 4
                # % (i, nof, img.min(), p1, p50, p99, img.max(), l.sum(), l.std())
            )
            p50s.append(p50)
            mps.append(mp)
            means.append(img.mean())
            s2s.append(s2)
            stds.append(img.std())
            # dm = cv2.subtract(img, d)
            # p1, p50, p99 = np.percentile(dm, (0.1, 50, 99.9))
            # LogHelper.logText(
            #     '%3d of %d, %4d-%4d-%4d-%4d-%5d, %.0f-%.0f'
            #     % (i, nof, dm.min(), p1, p50, p99, dm.max(), np.mean(dm), np.std(dm))
            # )
        LogHelper.logText(
            '%4d-%d-%4d #%4d-%4d' % (np.std(p50s), np.std(mps), np.std(means), np.std(s2s), np.std(stds))
        )


def calculate_masked_stats():
    plate_no = "59798"
    parsed = get_plate_files(plate_no)
    for w in ['w2']:
        files = filter(lambda f: f.wave == w[1], parsed)
        # accum = np.zeros((2160, 2160), dtype=np.uint32)
        # files = filter(lambda x: 's1' not in x and 's7' not in x, all_files)
        nof = len(files)
        for i, frame in enumerate(files[0:5], 1):
            LogHelper.logText(frame.fullpath)
            img = imread(frame.fullpath)
            t = filters.threshold_yen(img)
            b1 = img > t
            b2 = binary_erosion(b1, square(2))
            b3 = binary_dilation(b2, square(10))
            b4 = binary_closing(b3, square(3))
            imm = np.ma.masked_where(b4, img)
            mn, mx = np.percentile(imm, (1, 99))
            LogHelper.logText(
                '%3d of %d, %4d-%4d-%4d-%5d, %.0f-%.0f'
                % (i, nof, imm.min(), mn, mx, imm.max(), imm.mean(), imm.std())
            )
            im2 = imm.filled(int(imm.mean()))
            out_name = "{0}\\{5}-{1}{2}-{3}-{4}.tif".format(ROOT_DIR, frame.row, frame.column, frame.site, LogHelper.init_ts, frame.experiment)
            imsave(out_name, im2)
            # clip = img
            # accum += clip
        # filename = os.path.join(ROOT_DIR, w + '-' + LogHelper.init_ts + '-%s.tiff')
        # LogHelper.logText(filename)
        # np.save(filename % 'accum', accum)
        # avg_u = (accum / len(files)).astype(np.uint16)
        # imsave(filename % 'avg_u', avg_u)
        # smooth = median(avg_u, disk(20))
        # imsave(filename % 'smooth', smooth)


LPTH = 0.0000005


def calculate_ilum():
    parsed = get_plate_files("59839")
    for w in ['2']:
        files = filter(lambda f: f.wave == w, parsed)[0:30]
        nof = len(files)
        # files = filter(lambda x: 's1' not in x and 's7' not in x, files)
        img0 = imread(files[0].fullpath)
        mp = np.argmax(np.bincount(img0.flat))
        s2 = mp - img0.min()
        accum = np.zeros_like(img0, dtype=np.int32)
        accum_cnt = np.ones_like(img0, dtype=np.int32)
        thresh_w = np.uint16(filters.threshold_otsu(img0))
        prt = (img0 > thresh_w).sum() * 1.0 / len(img0.flat)
        if prt > 0.2:
            thresh_w = img0.mean() + 10 * img0.std()
        LogHelper.logText('{0}'.format(thresh_w))
        # ls = 0
        for i in range(nof):
            frame = files[i]
            img = imread(frame.fullpath)
            mp = np.argmax(np.bincount(img.flat))
            s2 = mp - img0.min()
            t = mp + s2
            # LogHelper.logText('%4d-%4d-%5d (%.0f)' % (img.min(), img.mean(), img.max(), img.std()))
            img[img >= t] = 0
            accum += img
            accum_cnt += (img != 0)
            # av = (accum / accum_cnt).astype(np.uint16)
            # avs = filters.laplace(av, 31)
            # s = avs.std()
            # ds = abs(s - ls)
            LogHelper.logText('%3d of %d w%s# %s %d' % (i+1, nof, w, frame.shortname(), t))
            # ls = s
            # if ds < LPTH:
            #     break

        stats_dir = os.path.join(ROOT_DIR, "%s-stats" % files[0].plate)
        try:
            os.mkdir(stats_dir)
        except WindowsError, e:
            assert(e.winerror == 183)  # 'Cannot create a file when that file already exists'
        filename = os.path.join(stats_dir, "%s-w%s-%%s.tif" % (LogHelper.init_ti, w))
        LogHelper.logText(filename)
        tifsave(filename % 'accum', accum)
        tifsave(filename % 'accum_cnt', accum_cnt)

        avg_u = (accum / accum_cnt).astype(np.uint16)
        tifsave(filename % 'avg_u', avg_u)
        smooth = filters.rank.mean(avg_u, disk(50))
        tifsave(filename % 'smooth', smooth)


def calculate_empty_stats():
    parsed = get_plate_files("empty")
    for w in ['w1']:
        files = filter(lambda f1: f1.wave == w[1], parsed)
        nof = len(files)
        stats = []
        for i, f in enumerate(files):
            img = imread(f.fullpath)
            h = np.unique(img)
            p0_1, p99_9 = np.percentile(img, (1, 99)).astype(np.uint16)
            mn = img.min()
            mx = img.max()
            mean = img.mean()
            std = img.std()
            st = [f.row, f.column, f.site, i, nof, mn, p0_1, p99_9, mx, mean, std, mx-p99_9, len(h)]
            stats.append(st + [f])
            LogHelper.logText('%s%s%s %3d of %d, %4d-%4d-%4d-%5d, %.2f-%.2f-%3d %d' % tuple(st))
        stats.sort(key=lambda s: s[8])
        s = stats[0]
        fn = s.pop()
        LogHelper.logText('%s%s%s %3d of %d, %4d-%4d-%4d-%5d, %.2f-%.2f-%3d %d' % tuple(s))
        LogHelper.logText(fn.filename)



def dilum():
    # ilum = imread(r"T:\NewImaging\w2-2018-03-15-16-02-09-smooth.tiff")
    parsed = get_plate_files("59798")
    for w in ['w2']:
        files = filter(lambda f: f.wave == w[1], parsed)
        for i, frame in enumerate(files[0:1], 1):
            img = imread(frame.fullpath)
            r1 = rescale_intensity(img, out_range=np.uint8).astype(np.uint8)
            mn = img.min()
            mx = img.max()
            mean = np.mean(img)
            std = np.std(img)
            img[img > (mean + std)] = mn
            r2 = rescale_intensity(img, in_range=(mn, mx), out_range=np.uint8).astype(np.uint8)
            s = np.stack((r1, r1, r2), 2)
            # img2 = (np.int32(img) - ilum)
            # img3 = np.clip(img2, 0, None)
            # img4 = rescale_intensity(img3, out_range=np.uint8).astype(np.uint8)
            out_name = "{0}\\{1}{2}-{3}-{4}.png".format(ROOT_DIR, frame.row, frame.column, frame.site, LogHelper.init_ts)
            imsave(out_name, s)
            LogHelper.logText("*************** {0:s} ***************".format(out_name))


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
    parsed = get_plate_files(plate_name)
    superdim = (DIMP*4*6, DIMP*4*10, 3)
    superframe = np.ones(superdim, dtype=np.uint8) * 255
    for f in parsed:
        LogHelper.logText(f.filename)
        s = int(f.site) - 1
        c = int(f.column) - 2
        r = (ord(f.row) - ord('B'))
        y = sDIM*(s % 4) + DIMP*4*c
        x = sDIM*(s / 4) + DIMP*4*r
        img = imread(f.fullpath)
        img = rescale(img, 0.2, multichannel=True, preserve_range=True)
        imgp = np.pad(img, ((0,4), (0,4), (0,0)), 'constant', constant_values=128)
        LogHelper.logText('{0} read to go {1}x{2}'.format(f.filename, x, y))

        superframe[x:x+imgp.shape[0], y:y+imgp.shape[1]] = imgp
        LogHelper.logText(f.filename + ' placed')

    imsave(ROOT_DIR + '\\59476super.png', superframe)


def fit_polynomial(pixel_data):
    '''Return an "image" which is a polynomial fit to the pixel data

    Fit the image to the polynomial Ax**2+By**2+Cxy+Dx+Ey+F

    pixel_data - a two-dimensional numpy array to be fitted
    '''
    x,y = np.mgrid[0:pixel_data.shape[0], 0:pixel_data.shape[1]]
    x2 = x*x
    y2 = y*y
    xy = x*y
    x2y = x*x*y
    y3 = y*y*y
    x3 = x*x*x
    y2x = y*y*x
    o  = np.ones(pixel_data.shape)
    a = np.stack([x.flat, y.flat, x2.flat, y2.flat, xy.flat, o.flat], 1)
    # a = np.stack([x.flat, y.flat, x2.flat, y2.flat, xy.flat, x2y.flat, y3.flat, x3.flat, y2x.flat, o.flat], 1)
    mean, std = pixel_data.mean(), pixel_data.std()
    # z = (pixel_data.flat - mean) / std
    z = pixel_data.flat
    coeffs, residuals, rank, s = scipy.linalg.lstsq(a, z)
    LogHelper.logText('\n{:.8f}x + {:.8f}y + {:.8f}x^2 + {:.8f}y^2 + {:.8f}xy + {:.8f}', *coeffs)
    # LogHelper.logText('\n{:.8f}x + {:.8f}y + {:.8f}x^2 + {:.8f}y^2 + {:.8f}xy + {:.8f}x^2y + {:.8f}y^3 + {:.8f}x^3 + {:.8f}xy^2 + {:.8f}', *coeffs)
    # output_pixels = np.sum([coeff * index for coeff, index in zip(coeffs, [x,y,x2,y2,xy, x2y, y3, x3, y2x, o])], 0)
    output_pixels = np.sum([coeff * index for coeff, index in zip(coeffs, [x,y,x2,y2,xy,o])], 0)
    smooth = filters.rank.mean(pixel_data, disk(50))
    coeffs2, residuals2, rank2, s2 = scipy.linalg.lstsq(a, smooth.flat)
    LogHelper.logText('\n{:.8f}x + {:.8f}y + {:.8f}x^2 + {:.8f}y^2 + {:.8f}xy + {:.8f}', *coeffs2)

    return output_pixels, mean, std


def fit_polynomial3(pixel_data):
    '''Return an "image" which is a polynomial fit to the pixel data

    Fit the image to the polynomial Ax**2+By**2+Cxy+Dx+Ey+F

    pixel_data - a two-dimensional numpy array to be fitted
    '''
    x,y = np.mgrid[0:pixel_data.shape[0], 0:pixel_data.shape[1]]
    x2 = x*x
    y2 = y*y
    xy = x*y
    x2y = x*x*y
    y3 = y*y*y
    x3 = x*x*x
    y2x = y*y*x
    o  = np.ones(pixel_data.shape)
    a = np.stack([x.flat, y.flat, x2.flat, y2.flat, xy.flat, x2y.flat, y3.flat, x3.flat, y2x.flat, o.flat], 1)
    mean, std = pixel_data.mean(), pixel_data.std()
    # z = (pixel_data.flat - mean) / std
    z = pixel_data.flat
    coeffs, residuals, rank, s = scipy.linalg.lstsq(a, z)
    LogHelper.logText('\n{:.8f}x + {:.8f}y + {:.8f}x^2 + {:.8f}y^2 + {:.8f}xy + {:.8f}x^2y + {:.8f}y^3 + {:.8f}x^3 + {:.8f}xy^2 + {:.8f}', *coeffs)
    output_pixels = np.sum([coeff * index for coeff, index in zip(coeffs, [x,y,x2,y2,xy, x2y, y3, x3, y2x, o])], 0)
    smooth = filters.rank.mean(pixel_data, disk(50))
    coeffs2, residuals2, rank2, s2 = scipy.linalg.lstsq(a, smooth.flat)
    LogHelper.logText('\n{:.8f}x + {:.8f}y + {:.8f}x^2 + {:.8f}y^2 + {:.8f}xy + {:.8f}x^2y + {:.8f}y^3 + {:.8f}x^3 + {:.8f}xy^2 + {:.8f}', *coeffs2)

    return output_pixels, mean, std


def fit():
    avg1 = imread(r"T:\NewImaging\59833-stats\20180319162439-w1-avg_u.tif")
    f,m,s = fit_polynomial3(avg1)
    tifsave(r"T:\NewImaging\59833-stats\w1-avg_u-fit3.tif", f)
    avg1 = imread(r"T:\NewImaging\59833-stats\20180319162439-w2-avg_u.tif")
    f,m,s = fit_polynomial(avg1)
    tifsave(r"T:\NewImaging\59833-stats\w2-avg_u-fit3.tif", f)


if __name__ == "__main__":
    # calculate_empty_stats()
    # calculate_ilum()
    fit()
    # calculate_stats()
    # calculate_masked_stats()
    # dilum()
    # find_min()
    # stitch()
