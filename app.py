import streamlit as st
from together import Together
import os
from io import BytesIO
import json

from pdf_utils import extract_text_from_pdf, count_tokens, ENCODER
from summarization_utils import summarize_document
from arxiv_utils import search_arxiv_papers

API_KEY = "tgp_v1_auHD1U7Mvm_VSPvfSZoVQ9m_woHoVtyU06DJ60ln-R0"

if not API_KEY:
    st.error("Error: Together AI API key is missing. Please replace the placeholder with your actual key.")
    st.stop()

client = Together(api_key=API_KEY)

AVAILABLE_MODELS = {
    "Llama 3.3 70B (Default)": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "Llama 3.1 70B (Newer, 128K)": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "DeepSeek V3 (General, 128K)": "deepseek-ai/DeepSeek-V3",
    "DeepSeek R1 Distill Llama 70B (Reasoning, 8K)": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Mixtral 8x7B Instruct (8K)": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

MODEL_CONTEXT_WINDOWS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": 8192,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 128000,
    "deepseek-ai/DeepSeek-V3": 128000,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 8192,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv_papers",
            "description": "Searches for academic papers on Arxiv based on a user's query. Use this tool when the user asks to look up research papers, articles, or academic studies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for academic papers (e.g., 'large language models', 'AI ethics', 'quantum computing')."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of results to return, default is 3.",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def check_and_add_message(role, content):
    if not st.session_state.messages or \
       st.session_state.messages[-1]["role"] != role or \
       st.session_state.messages[-1]["content"] != content:
        st.session_state.messages.append({"role": role, "content": content})

if "messages" not in st.session_state:
    st.session_state.messages = []
    check_and_add_message("assistant", "Hello! How can I assist you today?")

if "all_document_contents" not in st.session_state:
    st.session_state.all_document_contents = []
if "all_document_names" not in st.session_state:
    st.session_state.all_document_names = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS["Llama 3.3 70B (Default)"]
if "last_action" not in st.session_state: # NEW: Initialize last_action
    st.session_state.last_action = None

st.title("🧠 DocuMind AI")
st.markdown("Your intelligent companion for document analysis, Q&A, and insights.")
st.divider()

with st.expander("📁 Document & Model Controls", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents for analysis.",
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("AI Model Selection")
        current_model_label = next(key for key, value in AVAILABLE_MODELS.items() if value == st.session_state.selected_model)

        selected_model_label = st.selectbox(
            "Select AI Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(current_model_label),
            help=f"Currently selected: {current_model_label}\nContext Window: {MODEL_CONTEXT_WINDOWS.get(st.session_state.selected_model, 'N/A')} tokens",
            label_visibility="collapsed"
        )

        if selected_model_label != current_model_label:
            st.session_state.selected_model = AVAILABLE_MODELS[selected_model_label]
            st.rerun()

    with col3:
        st.subheader("Quick Actions")
        if st.button("📊 Summarize All", disabled=not st.session_state.all_document_contents, use_container_width=True):
            if st.session_state.all_document_contents:
                st.session_state.last_action = "summarize" # MODIFIED: Set last_action
                st.rerun()

        if st.button("🗑️ Clear All", disabled=not st.session_state.all_document_names, use_container_width=True):
            st.session_state.all_document_contents = []
            st.session_state.all_document_names = []
            st.session_state.messages = []
            check_and_add_message("assistant", "Hello! How can I assist you today?")
            st.rerun()

if st.session_state.all_document_names:
    total_tokens = sum(count_tokens(doc) for doc in st.session_state.all_document_contents)
    st.info(f"📄 **Loaded Documents:** {', '.join(st.session_state.all_document_names)} | 📊 **Statistics:** {len(st.session_state.all_document_contents)} files, {total_tokens:,} tokens total.")

if uploaded_files:
    current_uploaded_names = [file.name for file in uploaded_files]
    if set(current_uploaded_names) != set(st.session_state.all_document_names):
        st.session_state.all_document_contents = []
        st.session_state.all_document_names = []

        with st.spinner(f"🔄 Processing {len(uploaded_files)} PDF file(s)..."):
            for uploaded_file in uploaded_files:
                bytes_data = BytesIO(uploaded_file.getvalue())
                extracted_text = extract_text_from_pdf(bytes_data)
                if extracted_text:
                    st.session_state.all_document_contents.append(extracted_text)
                    st.session_state.all_document_names.append(uploaded_file.name)
                else:
                    st.error(f"❌ Could not extract text from '{uploaded_file.name}'.")

        if st.session_state.all_document_contents:
            total_docs = len(st.session_state.all_document_contents)
            total_tokens_all_docs = sum(count_tokens(doc) for doc in st.session_state.all_document_contents)
            success_msg = f"🎉 Successfully loaded {total_docs} document(s) with {total_tokens_all_docs:,} tokens! Ask me anything about your documents."
            check_and_add_message("assistant", success_msg)


chat_history_container = st.container(height=500)

with chat_history_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Type your question here...", key="main_chat_input")

# NEW: Inject 'summarize' prompt if the button was clicked
if st.session_state.last_action == "summarize":
    prompt = "summarize"
    st.session_state.last_action = None # Reset the action

if prompt:
    check_and_add_message("user", prompt)

    combined_document_text = "\n\n".join(st.session_state.all_document_contents)
    current_model_full_context_window = MODEL_CONTEXT_WINDOWS.get(st.session_state.selected_model, 8192)
    QA_CONTEXT_LIMIT = int(current_model_full_context_window * 0.75)

    with chat_history_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if "summarize" in prompt.lower() and combined_document_text:
                message_placeholder.markdown(f"📊 Analyzing and summarizing your documents using **{selected_model_label}**... Please wait ⏳")
                try:
                    summary = summarize_document(client, combined_document_text, model_name=st.session_state.selected_model)
                    full_response = summary
                except Exception as e:
                    full_response = f"❌ **Error during summarization**: {str(e)}\n\nPlease try again or select a different model."

            elif "summarize" in prompt.lower() and not combined_document_text:
                full_response = "📄 **No documents loaded!** Please upload some PDF documents first before asking me to summarize."

            else:
                api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                if combined_document_text:
                    doc_for_qa = combined_document_text
                    current_doc_tokens = count_tokens(doc_for_qa)

                    if current_doc_tokens > QA_CONTEXT_LIMIT:
                        st.warning(
                            f"⚠️ **Document Size Notice**: Your documents ({current_doc_tokens:,} tokens) exceed the optimal Q&A limit "
                            f"of **{QA_CONTEXT_LIMIT:,} tokens** for '{selected_model_label}'. Using truncated content for analysis."
                        )
                        doc_for_qa = ENCODER.decode(ENCODER.encode(doc_for_qa)[:QA_CONTEXT_LIMIT])

                    context_message = {
                        "role": "system",
                        "content": f"You are DocuMind AI, an intelligent document assistant. The user has provided document content below. Use this information to answer their questions accurately and helpfully. If the question isn't document-related, provide general knowledge assistance.\n\nYou also have access to an external tool named 'search_arxiv_papers' to look up academic papers. **ONLY use 'search_arxiv_papers' when the user explicitly asks you to find, look up, or search for research papers, articles, or studies on a specific topic.** Do NOT use this tool for general questions, greetings, or when the answer can be found in the provided documents.\n\n---\nDocument Content:\n{doc_for_qa}\n---\n"
                    }
                    api_messages = [context_message] + api_messages
                    api_messages = [context_message] + api_messages

                try:
                    message_placeholder.markdown("🤔 Thinking...")

                    chat_completion_response = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=api_messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        stream=False
                    )

                    response_message = chat_completion_response.choices[0].message
                    tool_calls = response_message.tool_calls

                    if tool_calls:
                        message_placeholder.markdown(f"🤖 Identifying relevant tools for your request...")
                        available_functions = {
                            "search_arxiv_papers": search_arxiv_papers,
                        }
                        
                        tool_call = tool_calls[0]
                        function_name = tool_call.function.name
                        function_to_call = available_functions.get(function_name)
                        
                        if function_to_call:
                            function_args = json.loads(tool_call.function.arguments)
                            message_placeholder.markdown(f"⚙️ Calling tool: `{function_name}` with arguments: `{function_args}`...")
                            function_response = function_to_call(**function_args)
                            
                            api_messages.append(response_message)
                            api_messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": function_response,
                                }
                            )
                            
                            message_placeholder.markdown(f"🤖 Processing tool output and generating a response...")
                            stream_response_after_tool = client.chat.completions.create(
                                model=st.session_state.selected_model,
                                messages=api_messages,
                                stream=True
                            )

                            for chunk in stream_response_after_tool:
                                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                    full_response += chunk.choices[0].delta.content
                                    message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)

                        else:
                            full_response = f"❌ Error: Tool '{function_name}' not found."
                            message_placeholder.markdown(full_response)
                    
                    else:
                        stream_response_direct = client.chat.completions.create(
                            model=st.session_state.selected_model,
                            messages=api_messages,
                            stream=True
                        )

                        for chunk in stream_response_direct:
                            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"🔴 **API Error**: {str(e)}")
                    full_response = "❌ I'm having trouble connecting to the AI service or processing your request. Please check your API key, try selecting a different model, or rephrase your query."

            check_and_add_message("assistant", full_response)
            st.rerun()