import streamlit as st
from core.agent import graph

st.title("Doc Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        # TODO: wire up real graph invocation
        result = {"final_answer": "placeholder", "rag_answers": [], "web_results": []}

    st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})
    st.chat_message("assistant").write(result["final_answer"])

    with st.expander("Agent trace"):
        st.json(result)
