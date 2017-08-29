# -*- coding: utf-8 -*-

from string import Template
import base64
import cv2
import numpy


__version__ = "0.2"


def renderer(img, fmt):
    if not isinstance(img, numpy.ndarray):
        return None

    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), 90] if (fmt == 'webp') else None

    r, buf = cv2.imencode(".%s" % fmt, img, params=encode_param)
    if not r:
        return None

    out = "data:image/{0};base64,{1}".format(fmt, base64.b64encode(buf))
    return out


def render_images(imgs, fmt="png", title=None):
    rendered = []

    for img in imgs:
        res = renderer(img, fmt)
        if res is None:
            continue
        else:
            rendered.append(res)

    return "".join(
        Template('<img download="$name" id="$name" src="$data_uri" />').substitute({
            "data_uri": data,
            "name": str(title or data[25:28]) + "." + str(fmt or "unk")
        }) for data in rendered)


class VisualRecord(object):
    def __init__(self, title, imgs, footnotes="", fmt="jpg"):
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
            fmt='png'
            # imgs = cv2.resize(imgs.astype(numpy.uint8), None, fx=0.5, fy=0.5)

        self.title = title
        self.fmt = fmt

        if imgs is None:
            imgs = []

        self.imgs = imgs

        if not isinstance(imgs, (list, tuple, set, frozenset)):
            self.imgs = [self.imgs]

        self.footnotes = footnotes


    def render_footnotes(self):
        if not self.footnotes:
            return ""

        return Template("<pre>$footnotes</pre>").substitute({
            "footnotes": self.footnotes
        })


    def __str__(self):
        t = Template("""
<h4>$title</h4>
<span style="white-space: nowrap">$imgs</span>
$footnotes
<hr/>""")

        return t.substitute({
            "title": self.title,
            "imgs": render_images(self.imgs, self.fmt, self.title),
            "footnotes": self.render_footnotes()
        })
