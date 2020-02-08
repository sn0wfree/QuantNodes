# coding=utf-8
class Status(object):
    @staticmethod
    def _check_status_code_(resp):
        if resp.status_code == 200:
            pass
        else:
            raise ValueError('resp code is not 200: {}'.format(resp.status_code))


if __name__ == '__main__':
    pass
