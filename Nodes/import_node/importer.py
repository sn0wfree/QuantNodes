# coding=utf-8
# my_importer.py
import sys
import urllib.request as urllib2
from importlib import abc
from importlib.machinery import ModuleSpec

test = 1


class UrlMetaFinder(abc.MetaPathFinder):
    def __init__(self, base_url):
        self._base_url = base_url

    def find_spec(self, fullname, path=None, target=None):
        if path is None:
            base_url = self._base_url
        else:
            # 不是原定义的url就直接返回不存在
            if not path.startswith(self._base_url):
                return None
            base_url = path

        try:
            loader = UrlMetaLoader(base_url)
            return ModuleSpec(fullname, loader, is_package=loader.is_package(fullname))
        except Exception as e:
            return None


class UrlMetaLoader(abc.SourceLoader):
    def __init__(self, base_url):
        self.base_url = base_url

    def get_code(self, fullname):
        f = urllib2.urlopen(self.get_filename(fullname))
        return f.read()

    def get_data(self, **kwargs):
        pass

    def get_filename(self, fullname):
        return self.base_url + fullname + '.py'


def _install_meta(address):
    """
    远程import python 脚本
    :param address:
    :return:
    """
    finder = UrlMetaFinder(address)
    sys.meta_path.append(finder)


def install_tools(address):
    _install_meta(address)


if __name__ == '__main__':
    install_tools('http://localhost:12856/')

    import importer

    print(importer.test)
    pass
