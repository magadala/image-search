import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/search")

st.title("🔎 Image Semantic Search Engine")
query = st.text_input("Enter your image search query:")
submit = st.button("Search")

if submit and query:
    with st.spinner("Searching..."):
        try:
            response = requests.post(API_URL, json={"text": query})
            if response.status_code == 200:
                results = response.json()
                if results:
                    cols = st.columns(5)
                    for i, result in enumerate(results):
                        with cols[i % 5]:
                            st.image(result["image_path"], caption=result["caption"], use_container_width=True)
                            st.markdown(f"**Explanation:** {result['explanation']}")
                            st.markdown(f"**CLIP Score:** {result['score']:.2f}")
                            if "caption_score" in result:
                                st.markdown(f"**Caption Score:** {result['caption_score']:.2f}")
                            if result["score"] < 0.3:
                                st.warning("⚠️ Low confidence match. Try a longer or more descriptive query.")
                else:
                    st.warning("No results found.")
            else:
                st.error(f"Error from API: {response.status_code}")
        except Exception as e:
            st.error(f"Request failed: {e}")
