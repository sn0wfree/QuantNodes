# coding=utf-8
# my_importer.py
import sys
import urllib.request as urllib2
from importlib import abc
from importlib.machinery import ModuleSpec

test = 2


class UrlMetaFinder(abc.MetaPathFinder):
    def __init__(self, base_url, suffix='.py'):
        self._base_url = base_url
        self.suffix = suffix

    def find_spec(self, fullname, path=None, target=None):
        if path is None:
            base_url = self._base_url
        else:
            # 不是原定义的url就直接返回不存在
            if isinstance(path, str):
                if not path.startswith(self._base_url):
                    return None
                else:
                    base_url = path
            else:
                return None

        try:
            loader = UrlMetaLoader(base_url, suffix=self.suffix)
            return ModuleSpec(fullname, loader, is_package=loader.is_package(fullname))
        except Exception as e:
            return None


class UrlMetaLoader(abc.SourceLoader):
    def __init__(self, base_url, suffix='.py'):
        self.base_url = base_url
        self.suffix = suffix

    def get_code(self, fullname):
        f = urllib2.urlopen(self.get_filename(fullname))
        return f.read()

    def get_data(self, **kwargs):
        pass

    def get_filename(self, fullname):
        return self.base_url + fullname + self.suffix


def _install_meta(address, suffix='.py'):
    """
    远程import python 脚本
    :param address:
    :return:
    """
    finder = UrlMetaFinder(address, suffix=suffix)
    sys.meta_path.append(finder)


def install_tools(address, suffix='.py'):
    _install_meta(address, suffix=suffix)


if __name__ == '__main__':
    install_tools('http://localhost:12857/', suffix='.cpython-36.pyc')
    import basic_node
    print(basic_node.__name__)
    pass
