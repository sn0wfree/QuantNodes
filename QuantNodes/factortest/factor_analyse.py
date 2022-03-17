# coding=utf-8

def analyse_performance(factor_setting, prepro_setting, execute_setting, **kwargs):
    result_factor_analyse = {}
    result_factor_analyse['factor_setting'] = factor_setting
    result_factor_analyse['prepro_setting'] = prepro_setting
    result_factor_analyse['execute_setting'] = execute_setting

    print("本次分析的因子是:" + factor_setting['name'])
    print("本次分析开始时间:" + str(prepro_setting['adjdate_beg']))
    print("本次分析结束时间:" + str(prepro_setting['adjdate_end']))

    if 'factor_data' in kwargs.keys():
        # 已经自定义导入dataframe的情况"""
        # factor_custom=kwargs['factor_data']
        factor_std = factor_utils.get_factor_std(factor_setting, prepro_setting, factor_data=kwargs['factor_data'])
    else:
        factor_std = factor_utils.get_factor_std(factor_setting, prepro_setting)
    result_factor_analyse['factor_std'] = factor_std

    if execute_setting["EffectAnalysis_ifrun"]:
        try:
            print("因子有效性分析...")  # 包括IC、分组检验等
            result_analyse_effect = analyse_effect(factor_std, prepro_setting, execute_setting)
            result_factor_analyse['analyse_effect'] = result_analyse_effect
        except Exception:
            print("因子有效性分析出错!将跳过此项测试")
            result_analyse_effect = {}

    if execute_setting["EffectBySample_ifrun"]:
        try:
            print('分样本有效性分析...')
            result_analyse_effect_bysample = analys_effect_bysample(factor_std, prepro_setting, execute_setting)
            result_factor_analyse['analyse_effect_bysample'] = result_analyse_effect_bysample
        except Exception:
            print("分样本有效性分析出错!将跳过此项测试")
            result_analyse_effect_bysample = {}

    if execute_setting["EffectBy_SizeInd_ifrun"]:
        if (prepro_setting["preprocess_ifind"] + prepro_setting["preprocess_ifrisk"]) > 0:
            print('市值行业分层打分：因子不需要进行行业和风险中性！将跳过此项测试')
            result_score_by_size_ind = {}
        else:
            try:
                print("市值行业分层打分...")
                result_score_by_size_ind = score_by_size_ind(factor_std, prepro_setting, execute_setting)
                result_factor_analyse['score_by_size_ind'] = result_score_by_size_ind
            except Exception:
                print("市值行业分层打分出错!将跳过此项测试")
                result_score_by_size_ind = {}

    if execute_setting["RiskAnalysis_ifrun"]:
        try:
            print("计算因子与风险因子的相关关系...")
            corr_all = corr_riskfactor(factor_std, execute_setting)
            result_factor_analyse['corr_all'] = corr_all
        except Exception:
            print("计算因子与风险因子的相关关系出错!将跳过此项测试")
            corr_all = {}
    return result_factor_analyse


if __name__ == '__main__':
    pass
