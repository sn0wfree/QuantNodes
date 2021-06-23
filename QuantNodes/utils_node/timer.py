# coding=utf-8
def timer(func):
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)

        end = time.time()
        print(func.__name__ + ' spend: ' + str(end - start))
        return res

    return wrapper


if __name__ == '__main__':
    @timer
    def test1(a):
        return a + 1
    test1(1)

    pass
