import PyPDF2
from io import BytesIO
import tiktoken # Import tiktoken for token counting
import os # Added for the example usage block

# Define the encoding for the Llama 3.3 model
# 'cl100k_base' is generally recommended for Llama models.
ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a string using the specified encoder."""
    return len(ENCODER.encode(text))

def extract_text_from_pdf(pdf_file_path_or_buffer):
    """
    Extracts text from a PDF file.

    Args:
        pdf_file_path_or_buffer: Path to the PDF file (string) or a file-like object (BytesIO).

    Returns:
        A string containing all the text extracted from the PDF.
        Returns an empty string if extraction fails.
    """
    text = ""
    try:
        if isinstance(pdf_file_path_or_buffer, str):
            with open(pdf_file_path_or_buffer, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() or ""
        elif isinstance(pdf_file_path_or_buffer, BytesIO):
            reader = PyPDF2.PdfReader(pdf_file_path_or_buffer)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
        else:
            print("Unsupported PDF input type. Please provide a file path or BytesIO object.")
            return ""

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    return text

def chunk_text(text: str, max_tokens: int, overlap: int = 200) -> list[str]:
    """
    Splits text into chunks of a specified maximum token length, with optional overlap.
    Aims to preserve sentence boundaries by splitting on '.'
    """
    if not text:
        return []

    tokens = ENCODER.encode(text)
    token_length = len(tokens)
    chunks = []
    current_start = 0

    while current_start < token_length:
        current_end = min(current_start + max_tokens, token_length)
        chunk_tokens = tokens[current_start:current_end]
        chunk_text_decoded = ENCODER.decode(chunk_tokens) # Use a different variable name to avoid conflict

        # Try to find a good splitting point (e.g., end of a sentence) if not at end of text
        if current_end < token_length:
            last_period_index = chunk_text_decoded.rfind('.')
            if last_period_index != -1 and last_period_index > len(chunk_text_decoded) - overlap:
                chunk_to_add = chunk_text_decoded[:last_period_index + 1]
                # Adjust current_start based on the actual cut point, for more precise overlap
                # This re-encoding and decoding is an approximation for token-aware slicing
                current_start += len(ENCODER.encode(chunk_to_add)) - overlap
            else:
                # Fallback: if no good split point near end, just use the max_tokens limit
                chunk_to_add = chunk_text_decoded
                current_start = current_end - overlap
        else:
            chunk_to_add = chunk_text_decoded
            current_start = current_end # Move to end if it's the last chunk

        chunks.append(chunk_to_add)

        # Safety break for infinite loop in edge cases (e.g., if overlap is too large or chunking logic goes wrong)
        if len(chunks) > 1 and current_start <= tokens.index(chunk_tokens[0]): # If not progressing
            print("Warning: Chunking likely stuck in a loop or document is too long. Breaking.")
            break
        if current_start >= token_length: # Ensure we eventually exit
            break
    return chunks


if __name__ == "__main__":
    # Example usage if you want to test this module separately
    # You'll need a sample.pdf in the same directory for this test to work.
    sample_pdf_path = "sample.pdf" # Replace with a path to a test PDF
    if os.path.exists(sample_pdf_path):
        print(f"Attempting to extract text from {sample_pdf_path}...")
        extracted_content = extract_text_from_pdf(sample_pdf_path)
        if extracted_content:
            print("\n--- Extracted Content Sample ---")
            print(extracted_content[:500]) # Print first 500 characters
            print(f"\nTotal tokens: {count_tokens(extracted_content)}")
            print("\n--- Chunking Test (first 2 chunks) ---")
            test_chunks = chunk_text(extracted_content, max_tokens=1000, overlap=100)
            for i, chk in enumerate(test_chunks[:2]):
                print(f"Chunk {i+1} ({len(chk)} chars, {count_tokens(chk)} tokens):\n{chk[:200]}...")
        else:
            print("No content extracted or an error occurred.")
    else:
        print(f"Please create a '{sample_pdf_path}' file for testing this module.")