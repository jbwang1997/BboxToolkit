'''Get image size without reading entire image into Mems.

Ackonwledge:
    This file is modified from imagesize.
    https://github.com/shibukawa/imagesize_py
    https://github.com/jerem/ImageSize
'''
import re
import struct
from xml.etree import ElementTree


def _convertToPx(value):
    matched = re.match(r"(\d+)(?:\.\d)?([a-z]*)$", value)
    if not matched:
        raise ValueError("unknown length value: %s" % value)

    length, unit = matched.groups()
    if unit == "":
        output = int(length)
    elif unit == "cm":
        output = int(length) * 96 / 2.54
    elif unit == "mm":
        output = int(length) * 96 / 2.54 / 10
    elif unit == "in":
        output = int(length) * 96
    elif unit == "pc":
        output = int(length) * 96 / 6
    elif unit == "pt":
        output = int(length) * 96 / 6
    elif unit == "px":
        output = int(length)
    else:
        raise ValueError("unknown unit type: %s" % unit)

    return output


def imsize(filepath):
    """Get image size

    Args:
        filepath (str): path to image.

    Returns:
        width, height (int, int)
    """
    height = -1
    width = -1

    with open(str(filepath), 'rb') as fhandle:
        head = fhandle.read(26)
        size = len(head)
        # handle JPEGs
        if size >= 2 and head.startswith(b'\377\330'):
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except struct.error:
                raise ValueError("Invalid JPEG file")
        # handle JPEG2000s
        elif size >= 12 and head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            fhandle.seek(48)
            try:
                height, width = struct.unpack('>LL', fhandle.read(8))
            except struct.error:
                raise ValueError("Invalid JPEG2000 file")
        # see png edition spec bytes are below chunk length then and finally the
        elif size >= 24 and head.startswith(b'\211PNG\r\n\032\n') and head[12:16] == b'IHDR':
            try:
                width, height = struct.unpack(">LL", head[16:24])
            except struct.error:
                raise ValueError("Invalid PNG file")
        # Maybe this is for an older PNG version.
        elif size >= 16 and head.startswith(b'\211PNG\r\n\032\n'):
            # Check to see if we have the right content type
            try:
                width, height = struct.unpack(">LL", head[8:16])
            except struct.error:
                raise ValueError("Invalid PNG file")
        # handle BMP
        elif size >= 26 and head.startswith(b'BM'):
            try:
                width, height = struct.unpack('<II', head[18:26])
            except struct.error:
                raise ValueError('Invalid BMP file')
        # handle big endian TIFF
        elif size >= 8 and head.startswith(b"\x4d\x4d\x00\x2a"):
            offset = struct.unpack('>L', head[4:8])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack(">H", fhandle.read(2))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack(">HHLL", fhandle.read(12))
                if tag == 256:
                    if datatype == 3:
                        width = int(data / 65536)
                    elif datatype == 4:
                        width = data
                    else:
                        raise ValueError("Invalid TIFF file: width column data type should be SHORT/LONG.")
                elif tag == 257:
                    if datatype == 3:
                        height = int(data / 65536)
                    elif datatype == 4:
                        height = data
                    else:
                        raise ValueError("Invalid TIFF file: height column data type should be SHORT/LONG.")
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        elif size >= 8 and head.startswith(b"\x49\x49\x2a\x00"):
            offset = struct.unpack('<L', head[4:8])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack("<H", fhandle.read(2))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack("<HHLL", fhandle.read(12))
                if tag == 256:
                    width = data
                elif tag == 257:
                    height = data
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        # handle GIFs
        elif size >= 10 and head[:6] in (b'GIF87a', b'GIF89a'):
            # Check to see if content_type is correct
            try:
                width, height = struct.unpack("<hh", head[6:10])
            except struct.error:
                raise ValueError("Invalid GIF file")
        # handle SVGs
        elif size >= 5 and head.startswith(b'<?xml'):
            try:
                fhandle.seek(0)
                root = ElementTree.parse(fhandle).getroot()
                width = _convertToPx(root.attrib["width"])
                height = _convertToPx(root.attrib["height"])
            except Exception:
                raise ValueError("Invalid SVG file")

    return width, height
