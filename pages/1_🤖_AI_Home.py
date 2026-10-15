# pages/1_🤖_AI_Home.py
import streamlit as st

st.title("🌐 Artificial Intelligence (AI): The Broadest Field")


st.info("This application mirrors your architectural hierarchy diagram: AI ⊃ ML ⊃ DL ⊃ NLP.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Characteristics")
    st.markdown("""
    * **Broadest field** in tech[cite: 1].
    * **Includes ML, DL, NLP** and many other sub-disciplines[cite: 1].
    * **Goal:** Achieve human-like intelligence in machines[cite: 1].
    """)

with col2:
    st.subheader("Navigation Guide")
    st.markdown("""
    Use the sidebar menu to drill down deeper into the nested layers:
    1. **Machine Learning (ML)** - The subset focusing on learning from data[cite: 1].
    2. **Deep Learning (DL)** - Neural networks with multiple layers[cite: 1].
    3. **NLP** - Human language understanding and generation[cite: 1].
    """)