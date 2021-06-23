# coding=utf-8
import numpy as np
import os
import pandas as pd

__ConfigPath__ = ''

AVAILABLE_CONFIG_TYPE = ['json']


class ConfigThings(object):

    @classmethod
    def _get_config(cls, config_file: str):
        config_type = config_file.split('.')[-1]
        if isinstance(config_file, str):
            if os.path.isfile(__ConfigPath__ + os.sep + config_file):
                if config_type in AVAILABLE_CONFIG_TYPE:
                    return getattr(cls, '_get_config_' + config_type)(__ConfigPath__ + os.sep + config_file)
                # return __ConfigPath__ + os.sep + config_file
            elif os.path.isfile(config_file):
                if config_type in AVAILABLE_CONFIG_TYPE:
                    return getattr(cls, '_get_config_' + config_type)(config_file)

                # return config_file
            else:
                raise ValueError(f'config_file is not found! got {config_file} please check config_file location!')
        else:
            raise ValueError(f'config_file is not found! got {config_file} please check config_file location!')

    @classmethod
    def _set_config(cls, config_path, obj, ):
        config_type = config_path.split('.')[-1]
        if config_type in AVAILABLE_CONFIG_TYPE:
            getattr(cls, '_set_config_' + config_type)(config_path, obj)
        else:
            raise ValueError(
                f'not supported config_type:{config_type}! current only accept {",".join(AVAILABLE_CONFIG_TYPE)}')

    @staticmethod
    def _set_config_json(config_path, obj):
        with open(config_path, 'w') as f:
            json.dump(obj, f)

    @staticmethod
    def _get_config_json(config_path):
        with open(config_path, 'r') as f:
            return json.dump(f)


# Quant Studio 系统对象
class QS_Object(ConfigThings):
    """Quant Studio 系统对象"""

    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._config = self._get_config(config_file)

        self._LabelTrait = {}
        self._ArgOrder = pd.Series()
        for iTraitName in self.visible_traits():
            iTrait = self.trait(iTraitName)
            if iTrait.arg_type is None: continue
            iLabel = (iTrait.label if iTrait.label is not None else iTraitName)
            iOrder = (iTrait.order if iTrait.order is not None else np.inf)
            self._LabelTrait[iLabel] = iTraitName
            self._ArgOrder[iLabel] = iOrder
        self._ArgOrder.sort_values(inplace=True)
        self.__QS_initArgs__()
        self._ConfigFile, Config = None, {}
        if config_file:
            if not os.path.isfile(config_file): config_file = __QS_ConfigPath__ + os.sep + config_file
            if os.path.isfile(config_file):
                self._ConfigFile = config_file
                with open(self._ConfigFile, "r", encoding="utf-8") as File:
                    FileStr = File.read()
                    if FileStr: Config = json.loads(FileStr)
        Config.update(sys_args)
        for iArgName, iArgVal in Config.items():
            if iArgName in self._ArgOrder.index: self[iArgName] = iArgVal
        self.trait_view(name="QSView",
                        view_element=View(*self.getViewItems()[0], buttons=[OKButton, CancelButton], resizable=True,
                                          title=getattr(self, "Name", "设置参数")))

    def __setstate__(self, state, trait_change_notify=False):
        return super().__setstate__(state, trait_change_notify=trait_change_notify)

    @property
    def ArgNames(self):
        return self._ArgOrder.index.tolist()

    @property
    def Args(self):
        return {iArgName: self[iArgName] for iArgName in self.ArgNames}

    @property
    def Logger(self):
        return self._QS_Logger

    def getViewItems(self, context_name=""):
        Prefix = (context_name + "." if context_name else "")
        Context = ({} if not Prefix else {context_name: self})
        return ([Item(Prefix + self._LabelTrait[iLabel]) for iLabel in self._ArgOrder.index], Context)

    def setArgs(self):
        Items, Context = self.getViewItems()
        if Context: return self.configure_traits(
            view=View(*Items, buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数"),
                      kind="livemodal"), context=Context)
        return self.configure_traits(
            view=View(*Items, buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数"),
                      kind="livemodal"))

    def getTrait(self, arg_name):
        return (self._LabelTrait[arg_name], self.trait(self._LabelTrait[arg_name]))

    def add_trait(self, name, *trait):
        Rslt = super().add_trait(name, *trait)
        iTrait = self.trait(name)
        if iTrait.arg_type is None: return Rslt
        iLabel = (iTrait.label if iTrait.label is not None else name)
        iOrder = (iTrait.order if iTrait.order is not None else np.inf)
        self._LabelTrait[iLabel] = name
        self._ArgOrder[iLabel] = iOrder
        self._ArgOrder.sort_values(inplace=True)
        return Rslt

    def remove_trait(self, name):
        if (name not in self.visible_traits()) or (self.trait(name).arg_type is None): return super().remove_trait(name)
        iLabel = self.trait(name).label
        Rslt = super().remove_trait(name)
        self._LabelTrait.pop(iLabel)
        self._ArgOrder.pop(iLabel)
        return Rslt

    def __iter__(self):
        return iter(self._LabelTrait)

    def __getitem__(self, key):
        return getattr(self, self._LabelTrait[key])

    def __setitem__(self, key, value):
        setattr(self, self._LabelTrait[key], value)

    def __delitem__(self, key):
        self.remove_trait(self._LabelTrait[key])

    def __QS_initArgs__(self):
        return None


if __name__ == '__main__':
    import json

    test = {'test': '1'}
    test2 = [1, '2', '3']
    with open('test.json', 'w') as f:
        json.dump(test2, f)
    pass
