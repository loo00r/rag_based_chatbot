import streamlit as st
from core.agent import graph

st.title("ПДР Асистент")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Запитайте про правила дорожнього руху..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.spinner("Думаю..."):
        result = graph.invoke({
            "query": query,
            "classification": "",
            "sub_queries": [],
            "rag_answers": [],
            "web_results": [],
            "final_answer": "",
            "iterations": 0,
        })

    st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})
    st.chat_message("assistant").write(result["final_answer"])

    with st.expander("Які пункти знайдено"):
        st.json({"rag_answers": result["rag_answers"], "web_results": result["web_results"]})
