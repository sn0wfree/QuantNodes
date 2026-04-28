# coding=utf-8
"""
QuantNodes Streamlit 应用入口

提供量化研究任务的 Web 界面，包括：
- 策略实验室：构建和回测策略
- 因子实验室：因子分析和管理
- 回测仪表盘：回测结果可视化
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="QuantNodes",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """主应用入口"""

    st.sidebar.title("📊 QuantNodes")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "选择功能",
        ["🏠 首页", "📈 策略实验室", "🔬 因子实验室", "📉 回测仪表盘", "⚙️ 设置"]
    )

    if page == "🏠 首页":
        show_home()
    elif page == "📈 策略实验室":
        show_strategy_lab()
    elif page == "🔬 因子实验室":
        show_factor_lab()
    elif page == "📉 回测仪表盘":
        show_backtest_dashboard()
    elif page == "⚙️ 设置":
        show_settings()


def show_home():
    """首页"""
    st.title("🏠 QuantNodes 量化研究平台")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 策略实验室")
        st.markdown("构建、测试和优化您的量化策略")
        if st.button("进入", key="home_strategy"):
            st.session_state.page = "strategy"
            st.rerun()

    with col2:
        st.markdown("### 🔬 因子实验室")
        st.markdown("分析和管理您的因子库")
        if st.button("进入", key="home_factor"):
            st.session_state.page = "factor"
            st.rerun()

    with col3:
        st.markdown("### 📉 回测仪表盘")
        st.markdown("可视化回测结果和性能指标")
        if st.button("进入", key="home_backtest"):
            st.session_state.page = "backtest"
            st.rerun()

    st.markdown("---")

    st.markdown("""
    ## 快速开始

    1. **连接数据库** - 在设置中配置 ClickHouse 或 DuckDB
    2. **加载因子** - 在因子实验室中加载您的因子数据
    3. **构建策略** - 使用策略实验室构建策略表达式
    4. **回测分析** - 运行回测并分析结果
    """)

    st.markdown("---")

    st.markdown("""
    ## 系统架构

    | 组件 | 说明 |
    |------|------|
    | **core/** | 核心节点架构、Pipeline 组合原语 |
    | **factor_node/** | 因子计算引擎、97+ 内置算子 |
    | **database_node/** | 多数据库支持 (ClickHouse, DuckDB, MySQL) |
    | **symbolic/** | SQL 表达式编译引擎 |
    | **backtest/** | 回测引擎 |
    | **ai/** | AI 策略生成器 |
    """)


def show_strategy_lab():
    """策略实验室页面"""
    st.title("📈 策略实验室")

    st.markdown("""
    ### 构建您的量化策略

    在下方输入策略表达式，支持以下语法：
    - 算术运算: `+`, `-`, `*`, `/`
    - 比较运算: `>`, `<`, `>=`, `<=`, `==`
    - 逻辑运算: `&`, `|`, `~`
    - 因子引用: `@factor_name`
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        expression = st.text_area(
            "策略表达式",
            value="@return > 0.01 & @volume > 1000000",
            height=150
        )

        if st.button("▶️ 运行回测"):
            if expression:
                st.info(f"回测表达式: {expression}")
            else:
                st.warning("请输入策略表达式")

    with col2:
        st.markdown("### 可用因子")
        st.code("""
@return     - 收益率
@volume     - 成交量
@price      - 价格
@turnover   - 成交额
        """)

    st.markdown("---")
    st.markdown("### 回测结果")

    st.info("回测功能正在开发中...")


def show_factor_lab():
    """因子实验室页面"""
    st.title("🔬 因子实验室")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 因子列表")
        factor_list = ["return", "volume", "price", "turnover", "vwap"]
        selected = st.multiselect("选择因子", factor_list, default=["return"])

    with col2:
        st.markdown("### 因子详情")
        if selected:
            st.json({f: {"type": "float", "description": f"因子 {f}"} for f in selected})

    st.markdown("---")

    st.markdown("### 因子分析")

    chart_type = st.selectbox("图表类型", ["时序图", "截面分布", "IC 分析"])

    if chart_type == "时序图":
        st.line_chart({"数据": [1, 2, 3, 4, 5]})
    elif chart_type == "截面分布":
        st.bar_chart({"分布": [1, 2, 3, 4, 3, 2, 1]})
    else:
        st.info("IC 分析正在开发中...")


def show_backtest_dashboard():
    """回测仪表盘页面"""
    st.title("📉 回测仪表盘")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric("年化收益率", "12.5%", delta="2.3%")
    with metrics_col2:
        st.metric("夏普比率", "1.85", delta="0.15")
    with metrics_col3:
        st.metric("最大回撤", "-8.2%", delta="-1.1%")
    with metrics_col4:
        st.metric("胜率", "58.3%", delta="3.2%")

    st.markdown("---")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["收益曲线", "回撤曲线", "月度收益"])

    with chart_tab1:
        st.line_chart({"收益": [100, 102, 101, 105, 110, 108, 112, 115]})

    with chart_tab2:
        st.area_chart({"回撤": [0, -1, -2, -1, -3, -5, -4, -8]})

    with chart_tab3:
        st.bar_chart({"月份": [1.2, -0.5, 2.1, 1.8, -0.3, 1.5]})


def show_settings():
    """设置页面"""
    st.title("⚙️ 设置")

    st.markdown("### 数据库配置")

    db_type = st.selectbox("数据库类型", ["ClickHouse", "DuckDB", "MySQL", "SQLite"])

    if db_type == "ClickHouse":
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=8123)
        database = st.text_input("Database", value="default")
        username = st.text_input("Username", value="default")

    elif db_type == "DuckDB":
        path = st.text_input("数据库路径", value="./data.duckdb")

    elif db_type == "MySQL":
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=3306)
        database = st.text_input("Database")

    st.markdown("---")

    st.markdown("### AI 配置")

    llm_provider = st.selectbox("LLM 提供商", ["OpenAI", "Anthropic", "本地模型"])
    if llm_provider == "OpenAI":
        api_key = st.text_input("API Key", type="password")
        model = st.selectbox("模型", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])

    st.markdown("---")

    if st.button("💾 保存设置"):
        st.success("设置已保存！")


if __name__ == "__main__":
    main()