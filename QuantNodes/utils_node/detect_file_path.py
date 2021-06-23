# coding=utf-8
import os
from glob import glob
from platform import system


def detect_file_full_path(conf_file: str = 'conn.ini'):
    if system() == 'Windows':
        sep = '\\'
    else:
        sep = '/'

    path1 = os.path.abspath('.')
    path2 = os.path.abspath(os.path.dirname(os.getcwd()))
    path3 = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    if path1 + sep + conf_file in glob(path1 + sep + '*'):
        target_path = path1 + sep
    elif path2 + sep + conf_file in glob(path2 + sep + '*'):
        target_path = path2 + sep
    elif path3 + sep + conf_file in glob(path3 + sep + '*'):
        target_path = path3 + sep
    else:
        raise FileNotFoundError('connot locate {}'.format(conf_file, __file__))
    return target_path + conf_file


if __name__ == '__main__':
    pass
