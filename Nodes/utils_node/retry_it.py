# coding=utf-8
import time


def conn_try_again(max_retries=5, default_retry_delay=1, expect_error=Exception):
    def _conn_try_again(function):
        RETRIES = 0
        # 重试的次数
        count = {"num": RETRIES}

        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except expect_error as err:
                if count['num'] < max_retries:
                    time.sleep(default_retry_delay)
                    count['num'] += 1
                    return wrapped(*args, **kwargs)
                else:
                    status = 'Error'
                    sel = 'Error'
                    raise Exception(err)

        return wrapped

    return _conn_try_again


if __name__ == '__main__':
    pass
