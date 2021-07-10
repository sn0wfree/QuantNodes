# coding=utf-8
import struct

TypeList = {
    "FFD8FF": "JPEG",
    "89504E47": "PNG",
    "47494638": "GIF",
    "89484446": 'HDF5',
    "53514C69": "SQLITE"
}

"""
JPEG (jpg)，文件头：FFD8FF
PNG (png)，文件头：89504E47
GIF (gif)，文件头：47494638
TIFF (tif)，文件头：49492A00
Windows Bitmap (bmp)，文件头：424D
CAD (dwg)，文件头：41433130
Adobe Photoshop (psd)，文件头：38425053
Rich Text Format (rtf)，文件头：7B5C727466
XML (xml)，文件头：3C3F786D6C
HTML (html)，文件头：68746D6C3E
Email [thorough only] (eml)，文件头：44656C69766572792D646174653A
Outlook Express (dbx)，文件头：CFAD12FEC5FD746F
Outlook (pst)，文件头：2142444E
MS Word/Excel (xls.or.doc)，文件头：D0CF11E0
MS Access (mdb)，文件头：5374616E64617264204A
WordPerfect (wpd)，文件头：FF575043
Postscript (eps.or.ps)，文件头：252150532D41646F6265
Adobe Acrobat (pdf)，文件头：255044462D312E
Quicken (qdf)，文件头：AC9EBD8F
Windows Password (pwl)，文件头：E3828596
ZIP Archive (zip)，文件头：504B0304
RAR Archive (rar)，文件头：52617221
Wave (wav)，文件头：57415645
AVI (avi)，文件头：41564920
Real Audio (ram)，文件头：2E7261FD
Real Media (rm)，文件头：2E524D46
MPEG (mpg)，文件头：000001BA
MPEG (mpg)，文件头：000001B3
Quicktime (mov)，文件头：6D6F6F76
Windows Media (asf)，文件头：3026B2758E66CF11
MIDI (mid)，文件头：4D546864
"""
# 字节码转16进制字符串
def bytes2hex(bytes):
    num = len(bytes)
    hexstr = u""
    for i in range(num):
        t = u"%x" % bytes[i]
        if len(t) % 2:
            hexstr += u"0"
        hexstr += t
    return hexstr.upper()


def filetype(filename, typeList=None):
    """
    read binary data to check file type!
    :param filename:
    :param typeList:
    :return:
    """
    ftype = 'unknown'
    tl = TypeList if typeList is None else typeList
    len_hcode = max(map(lambda x: len(x), tl.keys()))
    if len_hcode == 0:
        return ftype
    with open(filename, 'rb') as binfile:  # 必需二制字读取
        numOfBytes = len_hcode // 2  # 需要读多少字节
        binfile.seek(0)  # 每次读取都要回到文件头，不然会一直往后读取
        hbytes = struct.unpack_from("B" * numOfBytes, binfile.read(numOfBytes))  # 一个 "B"表示一个字节
        f_hcode = bytes2hex(hbytes)
    for hcode in tl.keys():
        if f_hcode[:len(hcode)] == hcode:
            ftype = tl[hcode]
            break

    return ftype


if __name__ == '__main__':
    # import pandas as pd
    # import numpy as np
    # import sqlite3
    #
    # for i in range(10):
    #     df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
    #
    #     with sqlite3.connect('test.sqlite') as conn:
    #         df.to_sql('test2', conn, if_exists='replace')
    # print(filetype('../../../../../AppData/Roaming/JetBrains/PyCharm2021.1/scratches/test.sqlite'))
    pass
