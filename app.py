from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# ============================
# 1) .env の読み込み
# ============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================
# 2) LangChain の LLM設定
# ============================
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.7,
)


# ============================
# 3) LLM呼び出し関数
# ============================
def call_llm(user_input: str, expert_type: str) -> str:

    # 専門家ごとにシステムメッセージを切り替え
    if expert_type == "心理カウンセラー":
        system_message = (
            "あなたはプロの心理カウンセラーです。"
            "相手の心情に寄り添い、優しく具体的なアドバイスをしてください。"
        )
    elif expert_type == "マーケティング専門家":
        system_message = (
            "あなたはプロのマーケティングコンサルタントです。"
            "論理的で実用的なマーケティング戦略を提案してください。"
        )
    else:
        system_message = "あなたは有能な専門家として、分かりやすく答えてください。"

    # LangChain プロンプト構築
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "{user_input}"),
        ]
    )

    chain = prompt | llm

    # LLMに問い合わせ
    response = chain.invoke({"user_input": user_input})

    # 返答（Assistantのメッセージ部分）
    return response.content


# ============================
# 4) Streamlit UI
# ============================

st.title(" LLMアプリ（LangChain × Streamlit）")

# アプリの説明表示
st.markdown(
    """
###  アプリの概要
このアプリでは、入力したテキストについて  
選択した **専門家の視点** から回答を生成します。

1. 専門家の種類を選ぶ  
2. テキストを入力  
3. 「送信」ボタンを押す  

すると、専門家としての回答が表示されます。
"""
)

# ラジオボタンで専門家を選択
expert_type = st.radio(
    "回答してほしい専門家を選んでください：",
    ("心理カウンセラー", "マーケティング専門家"),
)

# 入力フォーム
user_input = st.text_input("質問や相談内容を入力してください：")

# ボタン
if st.button("送信"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("AIが回答を生成しています..."):
            answer = call_llm(user_input, expert_type)
            st.success("回答が生成されました！")
            st.write("### 回答")
            st.write(answer)