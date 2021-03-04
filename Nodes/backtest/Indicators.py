class Statistics(object):
    @staticmethod
    def cash(obj, code):
        for trade in obj:
            if trade.code == code and trade.side == 'sell':
                yield trade.trade_result_trade_size

    @staticmethod
    def cash_values(obj):
        return sum([trade.trade_result_trade_size for trade in obj if trade.side == 'sell'])

    @classmethod
    def securities(cls, obj):

        return {code: sum(cls.share(code)) for code in obj.codes()}

    @staticmethod
    def security_cost(obj):
        return sum([trade.trade_result_cost_size for trade in obj if trade.side == 'buy'])

    @staticmethod
    def security_cost_without_fee(obj):
        return sum([trade.trade_result_trade_size for trade in obj if trade.side == 'buy'])

    @staticmethod
    def share(obj, code):
        for trade in obj:
            if trade.code == code and trade.side == 'buy':
                yield trade.traded_size


class Indicators(object):
    """
    计算回测结果的指标信息
    """
    pass
