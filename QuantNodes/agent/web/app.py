# coding=utf-8
"""
Web界面

基于 Streamlit 的 Agent Web 界面。
"""

import asyncio
import streamlit as st

from QuantNodes.agent import Agent


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None


def main():
    st.set_page_config(
        page_title="QuantNodes Agent",
        page_icon="📊",
        layout="wide"
    )

    init_session()

    st.title("📊 QuantNodes Agent")

    with st.sidebar:
        st.header("设置")
        workspace = st.text_input("工作目录", value="./workspace")
        model = st.selectbox(
            "模型",
            ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"]
        )

        if st.button("初始化Agent"):
            with st.spinner("初始化中..."):
                try:
                    st.session_state.agent = Agent(
                        workspace=workspace,
                        config={"model": model}
                    )
                    st.success("Agent 初始化成功！")
                except Exception as e:
                    st.error(f"初始化失败: {e}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("请输入您的策略需求..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.agent is None:
            with st.chat_message("assistant"):
                st.error("请先在侧边栏初始化 Agent")
            return

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    loop = asyncio.new_event_loop()
                    response = loop.run_until_complete(
                        st.session_state.agent.run(prompt)
                    )
                    loop.close()
                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                except Exception as e:
                    st.error(f"执行失败: {e}")


if __name__ == "__main__":
    main()
