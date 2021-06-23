# coding=utf-8
import unittest

from QuantNodes.conf_node.parse_rfc_1738_args import _parse_rfc1738_args


class MyTestCase(unittest.TestCase):
    def test_parse_url(self):
        c = _parse_rfc1738_args("clickhouse://test:sysy@199.199.199.199:1234/drre")
        real = {'name': 'clickhouse', 'username': 'test', 'password': 'sysy', 'port': 1234, 'database': 'drre',
                'host': '199.199.199.199'}
        self.assertDictEqual(real, c)


if __name__ == '__main__':
    unittest.main()
