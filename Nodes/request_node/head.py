# coding=utf-8
from functools import wraps

from Nodes.request_node.session import Session
from Nodes.request_node.status import Status

ACCEPT_METHODS = ['get', 'post']


class SpiderBotHead(Session, Status):
    def __init__(self, session=None, html_session_kwargs=None):
        """

        :param session:
        :param html_session_kwargs:
        """

        self.session = self._create_session_(session=session, html_session_kwargs=html_session_kwargs)

    def decorator(self):
        def get_func(func):
            @wraps(func)
            def _temp(*args, **kwargs):
                if 'session' in kwargs.keys():
                    if kwargs.get('session') is None:
                        kwargs['session'] = self.session
                    else:
                        pass
                return func(*args, **kwargs)
            return _temp
        return get_func

    def __call__(self, url, render=True, method='get'):
        if method in ACCEPT_METHODS:
            resp = getattr(self.session, method)(url)
            self._check_status_code_(resp)
            if render:
                resp.html.render()
            return resp
        else:
            raise ValueError('method only accept {}'.format(','.join(ACCEPT_METHODS)))

    def get(self, url, render=True):
        """

        :param url:
        :param render:
        :return:
        """

        return self.__call__(url, render=render, method='get')

    @classmethod
    def _get(cls, url, session, render=True):
        """

        :param url:
        :param session:
        :param render:
        :return:
        """
        # session = cls._create_session_(session=session, html_session_kwargs=html_session_kwargs)
        resp = session.get(url, )
        cls._check_status_code_(resp)
        if render:
            resp.html.render(retries=8)
        return resp

    def close(self):
        if hasattr(self.session, "_browser"):
            self.session.close()

    def __enter__(self):
        return self.session
        # if isinstance(self.url, str):
        #     resp = getattr(self.session, self.method)(self.url)
        #     resp.html.render()
        #     return resp
        # else:
        #     raise ValueError('url must be str')

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.close()


#
# class GET(SpiderBotHead):
#     @staticmethod
#     def get(url, render=True, **kwargs):
#         resp = SpiderBotHead().get(url, render=render)
#         return resp


if __name__ == '__main__':
    pass
