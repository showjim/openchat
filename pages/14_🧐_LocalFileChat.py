# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os, base64, gc, uuid, re
from typing import List

import openai, glob
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding #HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex, ServiceContext,
    SimpleDirectoryReader, StorageContext,
    load_index_from_storage)
from llama_index.core.vector_stores.simple import (
    DEFAULT_VECTOR_STORE,
    NAMESPACE_SEP,
)
from llama_index.vector_stores.faiss import FaissVectorStore

import streamlit as st
import faiss



# Default embedding dimensions - will be updated based on the actual model
DEFAULT_EMBED_DIM = 768  # Default for nomic-embed-text
faiss_index = None  # Will be initialized with correct dimensions
BASE_URL = 'http://127.0.0.1:11434/'


@st.cache_resource
def load_llm(model: str = "deepseek-r1:1.5b", temperature: float = 0.6):
    global BASE_URL
    llm = Ollama(model=model, request_timeout=120.0, temperature=temperature,base_url=BASE_URL)
    return llm


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def set_reload_flag():
    # st.write("New document need upload")
    st.session_state["LocalFileReloadFlag"] = True


def set_reload_db_flag():
    st.session_state["IndexReloadFlag"] = True


def get_all_files_list(source_dir, ext:str = "faiss"):
    all_files = []
    result = []
    all_files.extend(
        glob.glob(os.path.join(source_dir, f"*.{ext}"), recursive=False)
    )
    for filepath in all_files:
        file_name = Path(filepath).name
        if file_name.startswith("default_"):
            result.append(file_name)
    return result

@st.cache_resource
def load_single_vectordb(workDir:str, vsFileName:str):
    vsFilePath = os.path.join(workDir, f"{vsFileName}")
    # load index from disk
    vector_store = FaissVectorStore.from_persist_path(persist_path=vsFilePath)  # .from_persist_dir(workDir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=workDir
    )
    index = load_index_from_storage(storage_context=storage_context)
    return index

@st.cache_resource
def load_vectordbs(workDir:str, all_files: List[str]):
    """
    Load and merge multiple vector databases into a single index.
    This function properly preserves node embeddings when merging.
    """
    all_nodes = []
    
    try:
        for filename in all_files:
            index = load_single_vectordb(workDir, filename)
            
            # Get nodes from the index's document store
            nodes_dict = index.storage_context.docstore.docs
            
            # Get the vector store to access embeddings
            vector_store = index.storage_context.vector_store
            
            # Add each node to our collection, ensuring embeddings are preserved
            for doc_id, node in nodes_dict.items():
                # Try to get the embedding from the vector store if available
                try:
                    if hasattr(vector_store, 'get_embedding'):
                        node_embedding = vector_store.get_embedding(doc_id)
                        if node_embedding is not None:
                            node.embedding = node_embedding
                except:
                    # If we can't get the embedding, the node will be re-embedded later
                    pass
                
                all_nodes.append(node)
        
        # Create a new index with all collected nodes
        if all_nodes:
            final_index = VectorStoreIndex(nodes=all_nodes)
            return final_index
        else:
            st.error("No nodes found in the selected files.")
            return None
    except Exception as e:
        st.error(f"Error loading vector databases: {str(e)}")
        return None

def process_thinking_part(stream):
    """
    Process the thinking part of the streaming response.
    Handles the <think> tags and displays the thinking process.
    """
    full_response = ""
    thinking_content = ""
    in_thinking_block = False
    
    with st.status("Thinking...", expanded=True) as status:
        thinking_placeholder = st.empty()

        try:
            for chunk in stream:
                if chunk is None:
                    continue
                    
                # Check for thinking tags
                if '<think>' in chunk:
                    in_thinking_block = True
                    chunk = chunk.replace('<think>', '')
                
                if '</think>' in chunk:
                    in_thinking_block = False
                    chunk = chunk.replace('</think>', '')
                    status.update(label='Thinking complete', state='complete', expanded=False)
                    break
                
                if in_thinking_block or not ('<think>' in full_response or '</think>' in full_response):
                    thinking_content += chunk
                    thinking_placeholder.markdown(thinking_content + "▌")
                
                full_response += chunk
        except Exception as e:
            st.error(f"Error processing thinking stream: {str(e)}")
            
    thinking_placeholder.markdown(thinking_content)
    return thinking_content

def process_answer_part(stream):
    """
    Process the answer part of the streaming response.
    Displays the answer incrementally as it's received.
    """
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        for chunk in stream:
            if chunk is None:
                continue
                
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
    except Exception as e:
        st.error(f"Error processing answer stream: {str(e)}")
        
    message_placeholder.markdown(full_response)
    return full_response

def display_message(messages):
    for message in messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"]):
                with st.expander("See thinking"):
                    st.markdown(message["content"]["thinking"])
                st.markdown(message["content"]["answers"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def create_query_engine(index):
    # Create the query engine, where we use a cohere reranker on the fetched nodes
    query_engine = index.as_query_engine(streaming=True) #, similarity_top_k=3, verbose=True,) #response_mode="refine", "compact"(default), "tree_summarize"

    # ====== Customise prompt template ======
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, in the case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
    return query_engine

def get_ollama_model_list():
    """
    Get a list of available models from Ollama.
    Returns a default list if the connection fails.
    """
    try:
        client = openai.OpenAI(
            base_url=BASE_URL + 'v1/',
            # required but ignored
            api_key='ollama',
        )

        list_completion = client.models.list()
        models = [model.id for model in list_completion.data]
        return models
    except Exception as e:
        st.warning(f"Could not connect to Ollama at {BASE_URL}: {str(e)}")
        # Return some default models as fallback
        return ["llama2", "mistral", "deepseek-r1:1.5b"]

def main():
    if "id" not in st.session_state:
        st.session_state.id = uuid.uuid4()
        st.session_state.file_cache = {}
    if "OpenChatReloadFlag" not in st.session_state:
        st.session_state["OpenChatReloadFlag"] = True
    if "QueryEngine" not in st.session_state:
        st.session_state["QueryEngine"] = None
    if "IndexReloadFlag" not in st.session_state:
        st.session_state["IndexReloadFlag"] = True
    if "LocalVectorDB" not in st.session_state:
        st.session_state["LocalVectorDB"] = None

    session_id = st.session_state.id
    work_path = os.path.abspath('.')
    workDir = os.path.join(work_path, "workDir")

    with st.sidebar:
        st.header(f"RAG Setting")
        # select Ollama base url
        option = st.selectbox(
            "Select Base URL",
            ("http://localhost:11434/", "http://ollama:11434/", "http://127.0.0.1:11434/", "Another option..."),
        )


        global BASE_URL
        # Create text input for user entry
        if option == "Another option...":
            BASE_URL = st.text_input("Enter your other option...")
        else:
            BASE_URL = option

        models = get_ollama_model_list()
        model = st.selectbox(
            label="Select Model",
            options=models,
            index=0,
            on_change=set_reload_db_flag)

        # Setup LLM & embedding model
        try:
            llm = load_llm(model)
            Settings.llm = llm
            
            # Initialize embedding model
            embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest", base_url=BASE_URL)
            Settings.embed_model = embed_model
            
            # Initialize FAISS index with correct dimensions
            global faiss_index
            if faiss_index is None:
                # Get embedding dimension from the model if possible
                try:
                    # Create a test embedding to determine the dimension
                    test_embedding = embed_model.get_text_embedding("test")
                    embed_dim = len(test_embedding)
                    st.session_state["embed_dim"] = embed_dim
                    faiss_index = faiss.IndexFlatL2(embed_dim)
                    st.success(f"Initialized FAISS with dimension: {embed_dim}")
                except Exception as e:
                    # Fall back to default dimension
                    st.warning(f"Could not determine embedding dimension, using default: {DEFAULT_EMBED_DIM}")
                    st.session_state["embed_dim"] = DEFAULT_EMBED_DIM
                    faiss_index = faiss.IndexFlatL2(DEFAULT_EMBED_DIM)
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")

        st.header(f"Add your documents!")
        uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
        if st.button("Upload"):
            if uploaded_file:
                try:
                    # with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(workDir, uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write("Indexing your document...")

                    if os.path.exists(workDir):
                        loader = SimpleDirectoryReader(
                            # input_dir=temp_dir,
                            input_files=[file_path],
                            required_exts=[".pdf"],
                            recursive=False
                        )
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    docs = loader.load_data()

                    # Creating an index over loaded data
                    vector_store = FaissVectorStore(faiss_index=faiss_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)

                    # Store the index in local index file
                    ext = os.path.splitext(uploaded_file.name)
                    vsFileName = f"{ext[0]}.faiss"
                    vsFilePath = os.path.join(workDir, f"{DEFAULT_VECTOR_STORE}{NAMESPACE_SEP}{vsFileName}")
                    index.storage_context.persist(persist_dir=workDir, vector_store_fname=vsFileName)

                    # Inform the user that the file is processed and Display the PDF uploaded
                    st.success("PDF to Vector DB Done!")
                    # display_pdf(uploaded_file)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()
        # select the specified index base(s)
        index_file_list = get_all_files_list(workDir, "faiss")
        options = st.multiselect('2.What file do you want to exam?',
                                 index_file_list,
                                 on_change=set_reload_db_flag)
        if len(options) > 0:
            if st.session_state["IndexReloadFlag"] == True:
                with st.spinner('Load Index DB'):
                    index = load_vectordbs(workDir, options)
                    query_engine = create_query_engine(index)
                    st.session_state["QueryEngine"] = query_engine
                    st.session_state["IndexReloadFlag"] = False
            if (st.session_state["QueryEngine"] is not None):
                st.write("✅ " + ", ".join(options) + " Index DB Loaded")
        else:
            st.session_state["QueryEngine"] = None


    col1, col2 = st.columns([6, 1])

    # st.markdown("""
    #     # Agentic RAG powered by <img src="data:image/png;base64,{}" width="120" style="vertical-align: -3px;">
    # """.format(base64.b64encode(open("assets/crewai.png", "rb").read()).decode()), unsafe_allow_html=True)


    with col1:
        # st.header(f"🧐 Chat with Docs using Local LLMs")
        st.markdown("""
        ## 🧐 Local File Chat
        ##### RAG powered by <img src="data:image/png;base64,{}" width="25" style="vertical-align: -5px;">
    """.format(base64.b64encode(open("img/logo/ollama-logo.png", "rb").read()).decode()), unsafe_allow_html=True)

    with col2:
        st.button("Clear ↺", on_click=reset_chat)

    st.subheader("", divider='rainbow')

    # Initialize chat history
    if "messages" not in st.session_state:
        reset_chat()

    # Display chat messages from history on app rerun
    display_message(st.session_state.messages)

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if QueryEngine is available
        if st.session_state["QueryEngine"] is None:
            with st.chat_message("assistant"):
                error_message = "Please select at least one document to chat with first! Use the dropdown in the sidebar to select files."
                st.error(error_message)
                # Add error response to chat history
                st.session_state.messages.append({"role": "assistant", "content": {"answers": error_message, "thinking": "No documents selected."}})
        else:
            # Display assistant response in chat message container
            try:
                with st.chat_message("assistant"):
                    streaming_response = st.session_state["QueryEngine"].query(prompt)
                    message_thinking = process_thinking_part(streaming_response.response_gen)
                    message_answer = process_answer_part(streaming_response.response_gen)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": {"answers": message_answer, "thinking": message_thinking}})
            except Exception as e:
                with st.chat_message("assistant"):
                    error_message = f"An error occurred while processing your query: {str(e)}"
                    st.error(error_message)
                    # Add error response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": {"answers": error_message, "thinking": f"Error: {str(e)}"}})

if __name__ == "__main__":
    main()
