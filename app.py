from dotenv import load_dotenv
from main_page_modules import *
import streamlit as st
from sidebar_modules import *


def main():
    # load envrionment variables
    load_dotenv()

    # set default values
    if "pdf_docs" not in st.session_state:
        st.session_state["pdf_docs"] = []
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "pdfs_processed" not in st.session_state:
        st.session_state["pdfs_processed"] = False
    if "langchain_messages" not in st.session_state:
        st.session_state["langchain_messages"] = StreamlitChatMessageHistory()

    # set main page fundamentals
    set_main_page()

    # set sidebar settings module
    set_sidebar_settings_module()

    # set sidebar upload pdf module
    set_sidebar_upload_pdf_module()

    # get chain when pressing process button
    get_chain_when_pressing_process_button()

    # # set user input area if pdfs have been uploaded
    # set_input_area()

    # set sidebar actions module
    set_sidebar_actions_module()

    # set sidebar about project module
    set_sidebar_about_project_module()

    # set chat display module
    set_chat_display_module()

    sc = SocratesChain(
        prompt=PromptTemplate.from_template("tell us a joke about {topic}"),
        llm=ChatOpenAI(),
    )
    st.write(sc.run({"topic": "callbacks"}))


if __name__ == "__main__":
    main()
