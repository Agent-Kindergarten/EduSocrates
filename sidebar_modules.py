import os
import streamlit as st
from socrates_chain_utils import *


def set_sidebar_upload_pdf_module():
    with st.sidebar:
        st.header("Your documents")

        # clear pdf_docs and append all uploaded pdfs
        st.session_state["pdf_docs"].clear()
        st.session_state["pdf_docs"].extend(
            st.file_uploader(
                label="Upload your PDFs here and click on 'Process'",
                type="pdf",
                accept_multiple_files=True,
            )
        )


def set_sidebar_about_project_module():
    with st.sidebar:
        st.header("About")
        st.write(
            "This is a demo of EduSocrates, a chatbot that helps you review what you've learned."
        )
        st.write(
            "This is a project of [Agent Kindergarten](https://github.com/Agent-Kindergarten)."
        )
        st.write(
            "You can find the source code [here](https://github.com/Agent-Kindergarten/EduSocrates)."
        )


def set_sidebar_settings_module():
    with st.sidebar:
        st.header("Settings")
        st.write("You can change the following settings:")
        openai_api_key = st.text_input(label="- OpenAI API key", type="password")

        st.selectbox(
            label="- Select your model",
            options=("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"),
            index=0,
            key="model_name",
        )

        if st.button(label="Save settings"):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ Settings saved!")


def set_sidebar_actions_module():
    with st.sidebar:
        st.header("Actions")
        if st.button(label="Clear chat history", key="recently_cleared_chat_history"):
            st.session_state["langchain_messages"].clear()
            st.success("✅ Chat history cleared!")


def get_chain_when_pressing_process_button():
    with st.sidebar:
        if len(st.session_state["pdf_docs"]) == 0:
            st.button(label="Process", disabled=True)
        else:
            if st.button(label="Process", disabled=False, key="process_button"):
                with st.spinner("Processing"):
                    st.session_state["chain"] = get_chain_by_pdfs()
                if st.session_state["chain"] is None:
                    st.error("❌ Processing failed.")
                    return
                st.success("✅ PDFs have been processed!")
                st.session_state["pdfs_processed"] = True
