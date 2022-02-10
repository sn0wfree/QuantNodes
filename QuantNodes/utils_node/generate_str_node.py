# coding=utf-8
import hashlib
import random
import uuid


def random_str(num=6):
    uln = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    rs = random.sample(uln, num)  # 生成一个 指定位数的随机字符串
    a = uuid.uuid1()  # 根据 时间戳生成 uuid , 保证全球唯一
    b = ''.join(rs + str(a).split("-"))  # 生成将随机字符串 与 uuid拼接
    return b  # 返回随机字符串


def randon_str_hash(num=6):
    string = random_str(num=num)
    res = hashlib.md5(string.encode('utf-8')).hexdigest()
    return res


if __name__ == '__main__':
    randon_str_hash()
    pass
