import os
import shutil
import streamlit as st

from create_database import create_vector_database
from rag import ask_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc
        );
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
    }

    .status-card {
        padding: 1rem;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }

    .source-card {
        padding: 0.8rem;
        border-radius: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_uploaded" not in st.session_state:
    st.session_state.book_uploaded = False

if "book_name" not in st.session_state:
    st.session_state.book_name = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            📚 Book RAG Assistant
        </div>

        <div class="hero-subtitle">
            Upload a book and interact with it using
            Retrieval-Augmented Generation.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Book Library")

    uploaded_file = st.file_uploader(
        "Upload a PDF book",
        type=["pdf"],
        help="Upload a PDF document to create a searchable knowledge base."
    )

    if uploaded_file:

        st.success(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Process Book",
            use_container_width=True
        ):

            os.makedirs("uploads", exist_ok=True)

            pdf_path = os.path.join(
                "uploads",
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(
                "Reading book and creating vector database..."
            ):

                try:

                    create_vector_database(
                        pdf_path,
                        "chroma-db"
                    )

                    st.session_state.book_uploaded = True
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success(
                        "Book processed successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Error while processing book: {e}"
                    )

    st.divider()

    st.subheader("📊 Current Book")

    if st.session_state.book_uploaded:

        st.success(
            f"📕 {st.session_state.book_name}"
        )

        st.caption(
            "The book is ready for questions."
        )

    else:

        st.info(
            "Upload and process a book to start chatting."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Main Chat Area
# --------------------------------------------------

if not st.session_state.book_uploaded:

    st.info(
        "👈 Upload a PDF book from the sidebar and click "
        "**Process Book** to start."
    )

    st.markdown(
        """
        ### How it works

        **1. Upload**  
        Select a PDF book from your computer.

        **2. Process**  
        The application extracts the text, splits it
        into chunks and creates embeddings.

        **3. Retrieve**  
        Your question is compared with the book's
        vector database.

        **4. Generate**  
        DeepSeek generates an answer using the
        retrieved book context.

        **5. Chat**  
        Continue asking questions about the book.
        """
    )

else:

    st.subheader(
        f"💬 Chat with **{st.session_state.book_name}**"
    )

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander(
                    "📚 Retrieved pages"
                ):

                    for source in message["sources"]:

                        st.markdown(
                            f"""
                            <div class="source-card">
                                📄 Page {source}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat input

    query = st.chat_input(
        "Ask something about the book..."
    )

    if query:

        # User message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.markdown(query)

        # Assistant response

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the book..."
            ):

                try:

                    answer, docs = ask_question(
                        query
                    )

                    st.markdown(answer)

                    # Extract page numbers

                    pages = []

                    for doc in docs:

                        page = doc.metadata.get(
                            "page"
                        )

                        if page is not None:

                            pages.append(
                                page + 1
                            )

                    pages = sorted(
                        set(pages)
                    )

                    if pages:

                        with st.expander(
                            "📚 Retrieved pages"
                        ):

                            st.write(
                                ", ".join(
                                    f"Page {p}"
                                    for p in pages
                                )
                            )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": pages
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"Something went wrong: {e}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )