import os
import json
import numpy as np
import openai
import streamlit as st

# — CONFIGURATION —
OPENAI_MODEL = "o4-mini"
EMBED_MODEL  = "text-embedding-ada-002"
TOP_K        = 3
CHUNKS_DIR   = "chunks"
EMBED_FILE   = "embeddings.json"

PROMPT_HEADER = """You are an expert on Alex Hormozi’s 100M Offers. 
Below are the exact passages from the book. 
Answer the user’s question relying ONLY on that material. 
If the answer is not contained in the excerpts, say “I don’t see that in the provided material.”.

[BEGIN BOOK EXCERPTS]
"""
PROMPT_FOOTER = "\n[END BOOK EXCERPTS]\n\n[USER QUESTION]\n{}\n\n[ANSWER BELOW]\n"

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Missing OPENAI_API_KEY in environment.")
    st.stop()

# Load embeddings
with open(EMBED_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
filenames = [item["filename"] for item in data]
metadatas = [item["metadata"] for item in data]
vectors   = np.array([item["embedding"] for item in data])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

st.title("100M Offers © RAG Chat")
st.write("Type a question about 100M Offers and get a book-grounded answer.")

prompt = st.text_area("Enter your question:", height=120)
if st.button("Get Answer"):
    if not prompt.strip():
        st.warning("Please enter a question.")
    else:
        # Embed the question
        resp = openai.embeddings.create(model=EMBED_MODEL, input=prompt)
        q_vec = np.array(resp.data[0].embedding)

        # Retrieve top-K chunks
        scores = [(i, cosine_similarity(q_vec, vec)) for i, vec in enumerate(vectors)]
        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_K]

        # Assemble context
        context_blocks = []
        for idx, score in top_k:
            fname = filenames[idx]
            with open(os.path.join(CHUNKS_DIR, fname), "r", encoding="utf-8") as f:
                text = f.read().strip()
            context_blocks.append(f"### {metadatas[idx]}\n{text}\n")

        excerpt_text = "\n\n".join(context_blocks)
        full_prompt = PROMPT_HEADER + excerpt_text + PROMPT_FOOTER.format(prompt)

        # Call the model
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": full_prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Used Excerpts")
        for idx, score in top_k:
            st.write(f"• **{filenames[idx]}** (score: {score:.3f}) — {metadatas[idx]}")
