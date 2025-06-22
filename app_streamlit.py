# app_streamlit.py
import io
import streamlit as st
from main import main
import logging

# Add log handler
def get_log_stream():
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(handler)
    return log_stream, handler

# Global cache for embeddings
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
    st.session_state.document_processed = False

logger = logging.getLogger()

st.title("Basic chat - based on papers archive")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("Ask your question here"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    try:
        log_stream, handler = get_log_stream()

        if st.session_state.document_processed:
            logger.info("Using cached embeddings - faster processing!")
            embedded_docs, final_answer = main(
                query=query,
                log_stream=log_stream,
                cached_embeddings=st.session_state.embeddings
            )
        else:
            logger.info("First run - processing documents and creating embeddings")
            embedded_docs, final_answer = main(query=query, log_stream=log_stream)
            st.session_state.embeddings = embedded_docs
            st.session_state.document_processed = True
            logger.info("Embeddings cached for future queries")
        
        # Remove the handler so logs don't duplicate
        logging.getLogger().removeHandler(handler)
        # Show also logs 
        # Display logs in an expandable section
        with st.expander("Show logs"):
            st.text(log_stream.getvalue())

    except Exception as main_error:
        logging.error(f"Error in main: {str(main_error)}")
        final_answer = "Sorry, there was an error processing your request."

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    with st.chat_message("assistant"):
        st.write(final_answer)
    
