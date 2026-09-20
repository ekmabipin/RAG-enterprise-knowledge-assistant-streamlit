import streamlit as st

from src.rag_chain import RAGChain
from src.logger import get_logger


logger = get_logger(__name__)

@st.cache_resource
def get_rag_chain():
    return RAGChain()


st.set_page_config(
    page_title="NovaTech Knowledge Assistant",
    page_icon="💬",
    layout="centered",
)


st.title("💬 NovaTech Knowledge Assistant")

st.caption(
    "Ask questions about NovaTech company policies and employee information."
)


if "rag_chain" not in st.session_state:
    try:
        # st.session_state.rag_chain = RAGChain()
           
        logger.info("RAG chain initialized for Streamlit session.")
        

        st.session_state.rag_chain = get_rag_chain()

    except Exception:
        logger.exception(
            "Failed to initialize RAG chain for Streamlit session."
        )

        st.error(
            "The knowledge assistant could not be initialized. "
            "Please check the application configuration."
        )

        st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


if st.button("Clear conversation"):
    try:
        st.session_state.messages = []
        st.session_state.rag_chain.clear_memory()

        logger.info("Conversation cleared by user.")

        st.rerun()

    except Exception:
        logger.exception("Failed to clear conversation.")

        st.error(
            "The conversation could not be cleared. Please try again."
        )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Sources"):

                for source in message["sources"]:
                    st.markdown(f"- `{source}`")


query = st.chat_input(
    "Ask a question about NovaTech policies..."
)


if query:

    logger.info("Received user query from Streamlit.")

    logger.debug(
        "Streamlit user query=%r",
        query,
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching company knowledge..."):

            try:
                result = st.session_state.rag_chain.answer(query)

                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:

                    with st.expander("Sources"):

                        for source in sources:
                            st.markdown(f"- `{source}`")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

                logger.info(
                    "Streamlit query completed successfully. Sources=%d",
                    len(sources),
                )

            except Exception:

                logger.exception(
                    "Failed to process Streamlit user query."
                )

                error_message = (
                    "Sorry, I could not process your question. "
                    "Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )