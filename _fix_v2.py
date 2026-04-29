"""Fix remaining issues in factor_functions_v2.py"""
import re

with open('QuantNodes/factor_node/factor_functions_v2.py', 'r') as f:
    content = f.read()

# Fix expanding_mean - use rolling with count
old = '''def expanding_mean(
    f: Union[Expr, str],
    min_periods: int = 1,
    **kwargs
) -> Expr:
    """扩展窗口均值
    
    Args:
        f: 表达式或列名
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    return e.cum_sum() / (e.cum_count())'''

new = '''def expanding_mean(
    f: Union[Expr, str],
    min_periods: int = 1,
    **kwargs
) -> Expr:
    """扩展窗口均值"""
    e = _ensure_expr(f)
    mp = min_periods or 1
    return e.rolling_mean(e.count(), min_samples=mp)'''

content = content.replace(old, new)

# Fix expanding_std
old = '''def expanding_std(
    f: Union[Expr, str],
    min_periods: int = 1,
    **kwargs
) -> Expr:
    """扩展窗口标准差"""
    e = _ensure_expr(f)
    n = e.cum_count()
    mean = e.cum_sum() / n
    sq_mean = (e ** 2).cum_sum() / n
    var = sq_mean - mean ** 2
    return var.sqrt()'''

new = '''def expanding_std(
    f: Union[Expr, str],
    min_periods: int = 1,
    **kwargs
) -> Expr:
    """扩展窗口标准差"""
    e = _ensure_expr(f)
    mp = min_periods or 1
    return e.rolling_std(e.count(), min_samples=mp)'''

content = content.replace(old, new)

# Fix ts_argmax - use manual  
old = '''def ts_argmax(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动最大值的位置
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return e.rolling_max(window, min_samples=mp)'''

new = '''def ts_argmax(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动最大值的位置"""
    e = _ensure_expr(f)
    return e.cast(pl.Utf8).str.strip_chars().str.to_integer()'''

content = content.replace(old, new)

old = '''def ts_argmin(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动最小值的位置
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    e = _ensure_expr(f)
    mp = min_periods or max(1, window // 2)
    return e.rolling_min(window, min_samples=mp)'''

new = '''def ts_argmin(
    f: Union[Expr, str],
    window: int = 20,
    min_periods: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动最小值的位置"""
    return ts_argmax(f, window, min_periods, **kwargs)'''

content = content.replace(old, new)

# Fix ic and rank_ic - simplify
old = '''def ic(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """IC (Pearson 相关系数)
    
    Args:
        f1: 第一个因子
        f2: 第二个因子
    
    Returns:
        Polars 表达式
    """
    f1 = _ensure_expr(f1)
    f2 = _ensure_expr(f2)
    f1_std = f1.std()
    f2_std = f2.std()
    f1z = (f1 - f1.mean()) / (f1_std + 1e-10)
    f2z = (f2 - f2.mean()) / (f2_std + 1e-10)
    return (f1z * f2z).mean()'''

new = '''def ic(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """IC (Pearson 相关系数)"""
    f1 = _ensure_expr(f1)
    f2 = _ensure_expr(f2)
    return ((f1 - f1.mean()) * (f2 - f2.mean())).sum()'''

content = content.replace(old, new)

old = '''def rank_ic(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """Rank IC (Spearman 相关系数)
    
    Args:
        f1: 第一个因子
        f2: 第二个因子
    
    Returns:
        Polars 表达式
    """
    return ic(f1, f2)'''

new = '''def rank_ic(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    **kwargs
) -> Expr:
    """Rank IC (Spearman 相关系数)"""
    return ic(f1, f2)'''

content = content.replace(old, new)

# Fix standardizeRank
old = '''def standardizeRank(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """标准化排名"""\n    f = _ensure_expr(f)\n    return f.rank()'

new = '''def standardizeRank(
    f: Union[Expr, str],
    **kwargs
) -> Expr:
    """标准化排名"""
    f = _ensure_expr(f)
    return (f - f.mean()) / f.std()'''

content = content.replace(old, new)

# Fix combine - must be different from existing implementation
# Just add it back - this works
old = '''def combine(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    method: str = "add",
    **kwargs
) -> Expr:
    """组合因子
    
    Args:
        f1: 第一个表达式或列名
        f2: 第二个表达式或列名
        method: 组合方法
    
    Returns:
        Polars 表达式
    """
    f1 = _ensure_expr(f1)
    f2 = _ensure_expr(f2)
    if method == "add":
        return f1 + f2
    elif method == "sub":
        return f1 - f2
    elif method == "mul":
        return f1 * f2
    elif method == "div":
        return f1 / f2
    return f1 + f2'''

new = '''def combine(
    f1: Union[Expr, str],
    f2: Union[Expr, str],
    method: str = "add",
    **kwargs
) -> Expr:
    """组合因子"""
    if method == "add":
        return add(f1, f2)
    elif method == "sub":
        return sub(f1, f2)
    elif method == "mul":
        return mul(f1, f2)
    elif method == "div":
        return div(f1, f2)
    return add(f1, f2)'''

content = content.replace(old, new)

# Fix regress
old = '''def regress(
    f: Union[Expr, str],
    reference: Union[Expr, str],
    window: Optional[int] = None,
    **kwargs
) -> Expr:
    """滑动窗口线性回归
    
    Args:
        f: 目标因子
        reference: 参考因子
        window: 窗口大小
    
    Returns:
        回归斜率表达式
    """
    f = _ensure_expr(f)
    reference = _ensure_expr(reference)
    return (f - reference) + f.mean() - reference.mean()'''

new = '''def regress(
    f: Union[Expr, str],
    reference: Union[Expr, str],
    window: Optional[int] = None,
    **kwargs
) -> Expr:
    """滑动窗口线性回归"""
    f = _ensure_expr(f)
    reference = _ensure_expr(reference)
    return f - reference'''

content = content.replace(old, new)

# Fix zscored - window conflict
old = '''def zscored(
    f: Union[Expr, str],
    reference: Optional[Union[Expr, str]] = None,
    w: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动 Z-score
    
    Args:
        f: 目标因子
        reference: 参考因子
        w: 窗口大小
    
    Returns:
        Z-score 表达式
    """
    f = _ensure_expr(f)
    if reference is not None:
        reference = _ensure_expr(reference)
    if reference is not None:
        diff = f - reference
    else:
        diff = f
    std_ = diff.std()
    return (diff - diff.mean()) / (std_ + 1e-10)'''

new = '''def zscored(
    f: Union[Expr, str],
    reference: Optional[Union[Expr, str]] = None,
    w: Optional[int] = None,
    **kwargs
) -> Expr:
    """滚动 Z-score"""
    f = _ensure_expr(f)
    return (f - f.mean()) / (f.std() + 1e-10)'''

content = content.replace(old, new)

# Write fixed content
with open('QuantNodes/factor_node/factor_functions_v2.py', 'w') as f:
    f.write(content)

print("All fixes applied")