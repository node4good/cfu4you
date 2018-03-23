from datetime import datetime
import time
import traceback
import os
import sys
import logging
from vlogging import VisualRecord


def setup_file_handler(ts, logger):
    time_stamp = format(int((ts * 10) % 10000000), "6d")
    htmlfile = 'cfu4you' + time_stamp + '.html'
    fh = logging.FileHandler(htmlfile, mode="w")
    fh.propagate = False
    fh.old_close_ = fh.close

    def on_log_close():
        if 'last_type' in sys.__dict__:
            parts = ["file:/", os.getcwd().replace('\\', '/'), htmlfile]
            print '/'.join(parts)
        else:
            pass
            # webbrowser.open_new_tab(h)
        fh.old_close_()

    fh.close = on_log_close
    fh.stream.write('<style>body {white-space: pre; font-family: monospace;}</style>\n')
    logger.addHandler(fh)
    return fh


class LogHelper(object):
    OUTPUT_PREFIX = ''
    NO_LINE_TIME = False
    logger = logging.getLogger("cfu4you")
    logger.setLevel(logging.DEBUG)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', "%M:%S")
    _ch.setFormatter(_formatter)
    logger.addHandler(_ch)

    init_t = last_ts = time.time()
    init_ts = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.fromtimestamp(init_t))
    init_ti = "{:%Y%m%d%H%M%S}".format(datetime.fromtimestamp(init_t))
    _fh = None  # setup_file_handler(last_ts, logger)

    # Side effect
    logger.info(init_ts)



    @classmethod
    def write_to_html(cls, text):
        cls._fh and cls._fh.stream.write(text)


    @staticmethod
    def time(functor):
        t0 = time.time()
        i_ret = functor()
        t = time.time() - t0
        LogHelper.logText("{1:5}ms {0}", functor.func_name, int(t * 1000))
        return i_ret


    @classmethod
    def log_format_message(cls, msg):
        old = cls.last_ts
        cls.last_ts = time.time()
        stack_lines = traceback.extract_stack()
        good_lines = filter(lambda s: 'Helper' not in s[0], stack_lines)
        line_no = good_lines[-1][1]
        f_name = good_lines[-1][2]
        mark = "%s:%s" % (f_name, line_no)
        if cls.NO_LINE_TIME:
            d_msg = "{0:<20} | {1}".format(mark, msg)
        else:
            d = cls.last_ts - old
            d_msg = "[{0:4.0f}] {1:<20} {2}".format(d * 1000, mark, msg)
        return d_msg

    @classmethod
    def logText(cls, msg, *args):
        r_msg = msg.format(*args)
        d_msg = cls.log_format_message(r_msg)
        cls.logger.info(d_msg)
        return r_msg

    @classmethod
    def log(cls, msg, imgs, *args):
        t = cls.logText(msg, args)
        cls.logger.debug(VisualRecord(t, imgs))
        cls._fh and cls._fh.flush()
