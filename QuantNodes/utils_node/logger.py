# coding=utf-8

import logging
import sys, os


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# BASE_NAME = BASE_DIR.split(os.sep)[-1]
@Singleton
class LoggerHelper(object):
    def __init__(self, app_name=None, file_path="test.log", log_level=logging.INFO):
        if app_name is None:
            app_name = BASE_DIR.split(os.sep)[-1]
        else:
            app_name = str(app_name)
            # 获取logger实例，如果参数为空则返回root logger
        self.logger = logging.getLogger(app_name)

        # 指定logger输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

        # 文件日志
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter  # 也可以直接给formatter赋值

        # 为logger添加的日志处理器，可以自定义日志处理器让其输出到其他地方
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 指定日志的最低输出级别，默认为WARN级别
        self.logger.setLevel(log_level)

        self.warn = self.logger.warning
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.debug = self.logger.debug

    def sql(self, info: str):
        self.info("[SQL]: " + info)

    def status(self, info):
        self.info("[STATUS]： " + info)


Logger = LoggerHelper(file_path="test.log", log_level=logging.INFO)
if __name__ == '__main__':
    print(BASE_DIR, Logger)
    Logger.warn('test')
    Logger.info('test2')
    Logger.debug('test3')
    Logger.critical('test3')
    Logger.sql('test3')
    Logger.status('tst5')
    pass
