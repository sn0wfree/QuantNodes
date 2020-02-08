# coding=utf-8
import datetime

import requests_cache

now_ym = datetime.datetime.now().strftime("%Y-%m")
requests_cache.install_cache(f'{now_ym}.db')

from requests_html import HTMLSession


class Session(object):
    @staticmethod
    def _create_session_(session=None, html_session_kwargs=None):
        if session is None:
            if html_session_kwargs is None:
                html_session_kwargs = {}
            elif isinstance(html_session_kwargs, dict):
                pass
            else:
                raise ValueError('html_session_kwargs only accept dict or None!')
            session = HTMLSession(**html_session_kwargs)
        else:
            pass
            # request_node = request_node
        return session

    pass


if __name__ == '__main__':
    pass
