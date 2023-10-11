import os
import tempfile
import streamlit as st
from typing import List
from dotenv import load_dotenv
from deprecated import deprecated
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms.openai import OpenAI
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationKGMemory, ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.schema.vectorstore import VectorStore
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        if type(st.session_state["chain"]) is ConversationalRetrievalChain:
            self.status.update(
                label=f'**Context Retrieval:** {query} \n:warning: :orange[*Note that this retrieval question is generated based on the question you asked.*]'
            )
        else:
            self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def get_docs_from_pdfs() -> List[Document]:
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in st.session_state["pdf_docs"]:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    return docs


def get_document_chunks(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(docs)
    return document_chunks


def get_vectorstore(document_chunks: List[Document]) -> VectorStore:
    """
    This function first creates an embedding object,
    then returns a vectorstore from a list of documents.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embeddings)
    return vectorstore


def get_retriever(vectorstore: VectorStore) -> BaseRetriever:
    retriever = vectorstore.as_retriever()
    return retriever


def get_llm() -> BaseLLM:
    return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), streaming=True)


def get_chain_by_pdfs() -> Chain | None:
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("❌ Please set your OpenAI API key in the sidebar.")
        return

    # get documents
    documents = get_docs_from_pdfs()
    # split documents into chunks
    document_chunks = get_document_chunks(documents)
    vectorstore = get_vectorstore(document_chunks)
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    # Setup memory for contextual conversation
    msgs = st.session_state["langchain_messages"]
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=msgs, return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        retriever=retriever, llm=llm, memory=memory, verbose=True
    )
    return chain


def set_main_page():
    # set main page fundamentals
    st.set_page_config(page_title="EduSocrates", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)


@deprecated(reason="use set_chat_display_module() instead")
def set_input_area():
    st.header(":books: Talk to Socrates what you've just learned.")
    if len(st.session_state["pdf_docs"]) == 0:
        st.info("Upload your PDFs first.")
        return
    if st.session_state["pdfs_processed"] == False:
        st.info("Click on 'Process' first.")
        return

    user_question = st.text_input("Start your talk here:")
    if user_question == "":
        return
    if (
        user_question
        and st.session_state["chain"] is not None
        and st.session_state["recently_cleared_chat_history"] is False
    ):
        handle_userinput(user_question)


def handle_userinput(user_question: str):
    """This function handles user input."""
    response = st.session_state["chain"](user_question)
    st.write(response)


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
            "You can find the source code [here](https://github.com/Agent-Kindergarten/EduSocrates)"
        )


def set_sidebar_settings_module():
    with st.sidebar:
        st.header("Settings")
        st.write("You can change the following settings:")
        st.text_input(
            label="- OpenAI API key", type="password", key="webpage_openai_api_key"
        )
        if st.button(label="Save settings", key="webpage_save_settings"):
            os.environ["OPENAI_API_KEY"] = st.session_state["webpage_openai_api_key"]
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


def set_chat_display_module():
    st.header(":books: Talk to Socrates what you've just learned.")
    chain = st.session_state["chain"]
    if chain is None:
        st.info("Upload your PDFs first.")
        return

    msgs = st.session_state["langchain_messages"]
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello, I'm Socrates, and how can I help you?")

    avatars = {"human": "user", "ai": "🧙‍♂️"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    user_query = st.chat_input(placeholder="Ask me anything!")
    if user_query is not None:
        st.chat_message(name="user").write(user_query)
        if chain is not None:
            # response = chain(user_query)
            # st.chat_message(name="Socrates", avatar="🧙‍♂️").write(response["answer"])

            with st.chat_message(name="Socrates", avatar="🧙‍♂️"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                print(user_query, chain.input_keys)
                response = chain(
                    inputs={"question": user_query},
                    callbacks=[retrieval_handler, stream_handler],
                )


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


if __name__ == "__main__":
    main()
