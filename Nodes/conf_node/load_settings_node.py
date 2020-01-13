# coding=utf-8
import configparser

from Nodes.basic.basic_node import BasicNode


class SettingsLoader(object):
    @staticmethod
    def load_file(file_path, parser_func, **kwargs):
        with open(file_path, 'r') as f:
            return parser_func(f.read(), **kwargs)

    # @classmethod
    # def load_settings_from_yaml(cls, file_path, key):
    #     import yaml
    #     r = cls.load_file(file_path, yaml.load)
    #     return r[key]


class SettingsBaseNodes(BasicNode):
    def __init__(self, file_path, operator):
        super(SettingsBaseNodes, self).__init__(file_path)
        self.file_path = file_path
        self.operator = operator


class ConfigparserConfNode(SettingsBaseNodes):
    def __init__(self, file_path, section):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)
        super(ConfigparserConfNode, self).__init__(file_path, self.config)
        self.section = section

    def __getitem__(self, key):
        return self.operator.get(self.section, key)

    def __setitem__(self, key, value):
        self.operator.set(self.section, key, value)

    def items(self):
        return dict(self.operator.items(self.section))

    def keys(self):
        return self.operator.options(self.section)




if __name__ == '__main__':
    cpc = ConfigparserConfNode('/Users/sn0wfree/PycharmProjects/Nodes/Nodes/test/test.ini', 'test')
    print(cpc.keys())

    pass
