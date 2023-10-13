import deprecated
import os
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain


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
                label=f"**Context Retrieval:** {query} \n:warning: :orange[*Note that this retrieval question is generated based on the question you asked.*]"
            )
        else:
            self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def set_main_page():
    # set main page fundamentals
    st.set_page_config(page_title="EduSocrates", page_icon=":books:")
    # st.write(css, unsafe_allow_html=True)


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
