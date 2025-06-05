from together import Together
from typing import List
import tiktoken
from pdf_utils import count_tokens, chunk_text

# --- Define ENCODER locally for summarization_utils ---
ENCODER = tiktoken.get_encoding("cl100k_base")

# Map model API names to their actual context window sizes
MODEL_CONTEXT_WINDOWS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": 8192,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 128000,
    "deepseek-ai/DeepSeek-V3": 128000,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 8192,
}

# Max tokens for the response itself (summary)
MAX_NEW_TOKENS_SUMMARY = 700

# Function to get context window dynamically
def get_model_context_window(model_name: str) -> int:
    """Returns the total context window size for a given model."""
    return MODEL_CONTEXT_WINDOWS.get(model_name, 8192) # Default to 8K if not found

def summarize_document(client: Together, full_document_text: str, model_name: str) -> str:
    """
    Summarizes a potentially very long document by chunking it and summarizing each chunk,
    then summarizing the collected summaries if necessary.
    """
    if not full_document_text or not full_document_text.strip():
        return "No document content available to summarize."

    # Get the context window for the current model
    current_model_context_window = get_model_context_window(model_name)

    # Allocate some tokens for prompt overhead and desired summary length
    # A safe buffer to avoid exceeding context for prompt + chunk + response
    PROMPT_BUFFER_FOR_CHUNK = 500 # Tokens for instruction prompt + safety
    CHUNK_MAX_TOKENS = current_model_context_window - MAX_NEW_TOKENS_SUMMARY - PROMPT_BUFFER_FOR_CHUNK

    if CHUNK_MAX_TOKENS <= 0:
        return f"Model '{model_name}' has too small a context window ({current_model_context_window} tokens) to effectively summarize. Cannot create a valid chunk size."

    total_doc_tokens = count_tokens(full_document_text)
    print(f"Total document tokens: {total_doc_tokens}")
    print(f"Using model '{model_name}' with context window: {current_model_context_window} tokens.")


    if total_doc_tokens + PROMPT_BUFFER_FOR_CHUNK + MAX_NEW_TOKENS_SUMMARY <= current_model_context_window:
        print("Document fits in single context window. Summarizing directly.")
        return get_llama_summary_chunk(client, full_document_text, max_tokens=MAX_NEW_TOKENS_SUMMARY, model_name=model_name)

    print(f"Document too large ({total_doc_tokens} tokens). Chunking into segments of max {CHUNK_MAX_TOKENS} tokens...")
    chunks = chunk_text(full_document_text, max_tokens=CHUNK_MAX_TOKENS, overlap=100)
    print(f"Generated {len(chunks)} chunks.")

    if not chunks:
        return "Could not chunk the document for summarization."

    individual_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing part {i+1}/{len(chunks)} (tokens: {count_tokens(chunk)})...")
        chunk_tokens = count_tokens(chunk)

        # Recalculate available tokens for THIS chunk within the prompt
        available_for_chunk_in_prompt = current_model_context_window - PROMPT_BUFFER_FOR_CHUNK - MAX_NEW_TOKENS_SUMMARY
        if chunk_tokens > available_for_chunk_in_prompt:
            print(f"Warning: Chunk {i+1} ({chunk_tokens} tokens) is too large after initial chunking for its prompt. Truncating.")
            chunk = ENCODER.decode(ENCODER.encode(chunk)[:available_for_chunk_in_prompt])
            print(f"Truncated chunk {i+1} to {count_tokens(chunk)} tokens.")

        summary = get_llama_summary_chunk(client, chunk, max_tokens=MAX_NEW_TOKENS_SUMMARY, model_name=model_name)
        individual_summaries.append(summary)

    combined_summaries_text = "\n\n".join(individual_summaries)
    combined_summaries_tokens = count_tokens(combined_summaries_text)
    print(f"Combined summaries tokens: {combined_summaries_tokens}")

    if combined_summaries_tokens + PROMPT_BUFFER_FOR_CHUNK + MAX_NEW_TOKENS_SUMMARY <= current_model_context_window:
        print("Combined summaries fit in context. Final summarization.")
        return get_llama_summary_chunk(client, combined_summaries_text, max_tokens=MAX_NEW_TOKENS_SUMMARY, model_name=model_name)
    else:
        print("Combined summaries still too large. Performing recursive summarization.")
        # Recursive call: summarize the summaries, passing the model_name
        return summarize_document(client, combined_summaries_text, model_name=model_name)

def get_llama_summary_chunk(client: Together, text_to_summarize: str, max_tokens: int = MAX_NEW_TOKENS_SUMMARY, temperature: float = 0.3, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free") -> str:
    """
    Generates a summary of a single text chunk using the specified LLM model via Together AI.
    This is an internal helper function.
    """
    if not text_to_summarize.strip():
        return "No text provided to summarize for this chunk."

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the following document content concisely and accurately. Focus on key information and main ideas. Do not add conversational filler. Just the summary."},
        {"role": "user", "content": f"Please summarize the following document chunk:\n\n{text_to_summarize}"}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during chunk summarization with Together AI using model {model_name}: {e}")
        return f"Error summarizing chunk: {e}"