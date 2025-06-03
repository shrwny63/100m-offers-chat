import os
import json
import numpy as np
import openai
import streamlit as st

# —————————— CONFIGURATION ——————————

# Change this if you want a different Chat model (e.g., "o4-mini" or "o3")
OPENAI_MODEL = "o4-mini"

# The embedding model to use
EMBED_MODEL = "text-embedding-ada-002"

# How many chunks to retrieve each turn
TOP_K = 3

# Local folders/files
CHUNKS_DIR = "chunks"
EMBED_FILE = "embeddings.json"

# System prompt templates
SYSTEM_PROMPT_TEMPLATE = """You are an expert on Alex Hormozi’s 100M Offers.
Below are the exact passages from the book. Answer the user’s question relying ONLY on that material.
If the answer is not contained in the excerpts, say “I don’t see that in the provided material.”.

[BEGIN BOOK EXCERPTS]
{context}
[END BOOK EXCERPTS]
"""

# ———————— END CONFIGURATION ————————


# —————————— SETUP ——————————

# Ensure the OpenAI API key is available
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ Missing OPENAI_API_KEY in environment or Streamlit secrets.")
    st.stop()

# Load all precomputed embeddings once at startup
with open(EMBED_FILE, "r", encoding="utf-8") as f:
    embed_data = json.load(f)

filenames = [item["filename"] for item in embed_data]
metadatas = [item["metadata"] for item in embed_data]
vectors   = np.array([item["embedding"] for item in embed_data])


# Cosine-similarity helper
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Retrieval: given a question, return top-K (index, similarity) tuples
def retrieve_top_k(question: str, k: int = TOP_K):
    resp = openai.embeddings.create(model=EMBED_MODEL, input=question)
    q_vec = np.array(resp.data[0].embedding)
    scores = [(idx, cosine_similarity(q_vec, vec)) for idx, vec in enumerate(vectors)]
    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return top_k


# Build context text from chunk indices
def build_context(chunks_idx_list):
    blocks = []
    for idx, score in chunks_idx_list:
        fname = filenames[idx]
        with open(os.path.join(CHUNKS_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()
        blocks.append(f"### {metadatas[idx]}\n{text}")
    return "\n\n".join(blocks)


# —————————— APP UI ——————————

st.set_page_config(page_title="100M Offers Chat", page_icon="📚", layout="centered")
st.title("📖 100M Offers Chat (Conversational RAG)")

st.markdown(
    """
    **How it works:**  
    1. Type a message below as if you were talking to a chatbot.  
    2. Under the hood, we embed your new question, retrieve the top 3 relevant passages from *100M Offers*,  
       and send them plus the chat history to OpenAI.  
    3. The model replies using only those selected book excerpts.  
    """
)


# Initialize chat history if not present
if "messages" not in st.session_state:
    # Each entry is a dict: { "role": "user"/"assistant", "content": str }
    st.session_state.messages = []


# Display all previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:  # assistant
        with st.chat_message("assistant"):
            st.markdown(msg["content"])


# Accept new user input
if prompt := st.chat_input("Type your message here..."):
    # 1. Add the user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Retrieve top-K book chunks
    top_k = retrieve_top_k(prompt, TOP_K)
    context_text = build_context(top_k)

    # 3. Build the system prompt with those excerpts
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)

    # 4. Assemble the messages for OpenAI: first the system prompt, then the entire chat history
    messages_to_send = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        # copy user/assistant roles directly
        messages_to_send.append({"role": m["role"], "content": m["content"]})

    # 5. Call OpenAI Chat Completion
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_to_send
    )
    assistant_reply = response.choices[0].message.content.strip()

    # 6. Add the assistant reply to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # 7. Display the assistant’s reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    # Optionally show which chunks were used (for transparency)
    st.markdown("---")
    st.markdown("**Used Book Excerpts:**")
    for idx, score in top_k:
        fname = filenames[idx]
        meta = metadatas[idx]
        st.markdown(f"- `{fname}` (score {score:.3f}) — {meta}")
