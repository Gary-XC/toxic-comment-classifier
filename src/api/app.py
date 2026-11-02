# src/api/app.py

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/invocations"

st.set_page_config(page_title="Trust & Safety ML", page_icon="🛡️", layout="centered")

st.title("Multilingual Abusive Language Detection")
st.markdown("---")
st.write("Enter text in **English, French, Spanish, or Mandarin** to test the zero-shot moderation API.")

text_input = st.text_area("User Comment:", height=100, placeholder="Type a comment here...")

# T&S Operations threshold slider
threshold = st.slider("Manual Review Confidence Threshold", min_value=0.5, max_value=0.99, value=0.75)

if st.button("Analyze Payload", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Calling API..."):
            try:
                response = requests.post(API_URL, json={"text": text_input})
                response.raise_for_status()
                data = response.json()
                
                confidence = data["confidence"]
                latency = data["latency_ms"]
                
                st.markdown("### Results")
                col1, col2, col3 = st.columns(3)
                
                # Business Logic Routing
                if confidence >= threshold:
                    col1.error("Action: AUTO-DELETE")
                elif confidence >= 0.5:
                    col1.warning("Action: HUMAN REVIEW")
                else:
                    col1.success("Action: ALLOW")
                    
                col2.metric("Toxicity Probability", f"{confidence:.1%}")
                col3.metric("API Latency", f"{latency:.0f} ms")
                
                with st.expander("View Raw JSON Payload"):
                    st.json(data)
                    
            except Exception as e:  # noqa: BLE001
                st.error(f"API Connection Error: Ensure FastAPI is running on port 8080. ({e})")