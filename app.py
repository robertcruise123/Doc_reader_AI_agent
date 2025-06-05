import streamlit as st
from together import Together
import os
from io import BytesIO

# Import our utility functions (assuming these are in pdf_utils.py and summarization_utils.py)
# Ensure 'pdf_utils.py' and 'summarization_utils.py' are in the same directory as this script.
from pdf_utils import extract_text_from_pdf, count_tokens, ENCODER
from summarization_utils import summarize_document

# --- Configuration ---
API_KEY = "tgp_v1_auHD1U7Mvm_VSPvfSZoVQ9m_woHoVtyU06DJ60ln-R0" # <--- PASTE YOUR TOGETHER AI API KEY HERE

if not API_KEY:
    st.error("Error: Together AI API key is missing. Please replace the placeholder with your actual key.")
    st.stop()

# Initialize the Together client once
client = Together(api_key=API_KEY)

# --- Define available models and their context windows ---
AVAILABLE_MODELS = {
    "Llama 3.3 70B (Default)": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "Llama 3.1 70B (Newer, 128K)": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", # 128K context
    "DeepSeek V3 (General, 128K)": "deepseek-ai/DeepSeek-V3", # 128K context
    "DeepSeek R1 Distill Llama 70B (Reasoning, 8K)": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", # Typically 8K context
    "Mixtral 8x7B Instruct (8K)": "mistralai/Mixtral-8x7B-Instruct-v0.1", # 8K context
}

# Map model API names to their actual context window sizes
MODEL_CONTEXT_WINDOWS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": 8192,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 128000,
    "deepseek-ai/DeepSeek-V3": 128000,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 8192,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 8192,
}

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to hide Streamlit's default footer and header (optional, but makes it cleaner)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- Helper function to add messages without duplication ---
def check_and_add_message(role, content):
    """Adds a message to st.session_state.messages only if it's not a duplicate of the last message."""
    if not st.session_state.messages or \
       st.session_state.messages[-1]["role"] != role or \
       st.session_state.messages[-1]["content"] != content:
        st.session_state.messages.append({"role": role, "content": content})

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial assistant message only once per session
    check_and_add_message("assistant", "Hello! How can I assist you today?")

if "all_document_contents" not in st.session_state:
    st.session_state.all_document_contents = []
if "all_document_names" not in st.session_state:
    st.session_state.all_document_names = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_MODELS["Llama 3.3 70B (Default)"]

# --- App Title and Description ---
st.title("🧠 DocuMind AI")
st.markdown("Your intelligent companion for document analysis, Q&A, and insights.")
st.divider() # Simple visual separator

# --- File Upload and Model Selection Section ---
# Use an expander for file operations for a cleaner look when not needed
with st.expander("📁 Document & Model Controls", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents for analysis.",
            label_visibility="collapsed" # Hide default label for cleaner look
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
            st.rerun() # Necessary to update model and context details

    with col3:
        st.subheader("Quick Actions")
        # Ensure buttons are disabled if no documents are loaded
        if st.button("📊 Summarize All", disabled=not st.session_state.all_document_contents, use_container_width=True):
            if st.session_state.all_document_contents:
                # Add summarize request to chat history and trigger rerun for AI response
                check_and_add_message("user", "summarize")
                st.rerun() # Rerun to process the new prompt

        if st.button("🗑️ Clear All", disabled=not st.session_state.all_document_names, use_container_width=True):
            st.session_state.all_document_contents = []
            st.session_state.all_document_names = []
            st.session_state.messages = []
            check_and_add_message("assistant", "Hello! How can I assist you today?") # Reset initial message
            st.rerun() # Force rerun to clear display

# --- Document Status Display ---
if st.session_state.all_document_names:
    total_tokens = sum(count_tokens(doc) for doc in st.session_state.all_document_contents)
    st.info(f"📄 **Loaded Documents:** {', '.join(st.session_state.all_document_names)} | 📊 **Statistics:** {len(st.session_state.all_document_contents)} files, {total_tokens:,} tokens total.")

# --- File Upload Processing ---
# This block handles the actual processing of uploaded files and adding the success message.
if uploaded_files:
    current_uploaded_names = [file.name for file in uploaded_files]
    # Only process if new files are detected to prevent re-adding messages on unrelated reruns
    if set(current_uploaded_names) != set(st.session_state.all_document_names):
        st.session_state.all_document_contents = [] # Clear old content before adding new
        st.session_state.all_document_names = [] # Clear old names before adding new

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
            # No st.rerun() here - Streamlit will rerun naturally due to session_state change from message or file upload.


# --- Chat Interface (main display) ---
# This container will hold all chat messages.
chat_history_container = st.container(height=500) # Give it a fixed height with scrollbar for chat history

with chat_history_container:
    # Display all messages in history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Chat Input (Fixed at Bottom) ---
# This input should be outside the chat_history_container to stay at the bottom.
prompt = st.chat_input("Type your question here...", key="main_chat_input")

# --- Handle User Input and Generate AI Response ---
if prompt: # This block executes when a user submits a prompt
    check_and_add_message("user", prompt) # Add user's message to history

    # Prepare document context for AI
    combined_document_text = "\n\n".join(st.session_state.all_document_contents)
    current_model_full_context_window = MODEL_CONTEXT_WINDOWS.get(st.session_state.selected_model, 8192)
    QA_CONTEXT_LIMIT = int(current_model_full_context_window * 0.75)

    # Generate AI response within a placeholder to show streaming
    with chat_history_container: # Output the AI response directly into the chat history container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Handle summarization requests
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
                # General chat/Q&A logic
                # Pass full message history to the API for context
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

                    # Add document content as a system message for context
                    context_message = {
                        "role": "system",
                        "content": f"You are DocuMind AI, an intelligent document assistant. The user has provided document content below. Use this information to answer their questions accurately and helpfully. If the question isn't document-related, provide general knowledge assistance.\n\n---\nDocument Content:\n{doc_for_qa}\n---\n"
                    }
                    api_messages = [context_message] + api_messages

                try:
                    message_placeholder.markdown("🤔 Thinking...") # Indicate thinking process

                    stream_response = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=api_messages,
                        stream=True
                    )

                    for chunk in stream_response:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌") # Show typing indicator
                    message_placeholder.markdown(full_response) # Final response without indicator

                except Exception as e:
                    st.error(f"🔴 **API Error**: {str(e)}")
                    full_response = "❌ I'm having trouble connecting to the AI service. Please check your API key and try again."

            check_and_add_message("assistant", full_response) # Add assistant response to history
            st.rerun() # Necessary to clear the chat input box and update the chat history display
