# Doc_reader_AI_agent

An AI agent which reads the uploaded documents , can handle multiple files and give detailed answers based on the questions asked.
The projec6 directory looks like:
- app.py: The main Streamlit application file.
- pdf_utils.py: Contains functions for PDF text extraction and token counting.
- summarization_utils.py: Contains the summarize_document function for interacting with the LLM for summarization.
- requirements.txt (Optional but recommended): A file listing all Python dependencies (streamlit, together, pypdf, tiktoken). You can generate this using pip freeze     requirements.txt after installing dependencies.

First:
- you need to instal the libraries from the requirments.txt
  
Second:
- you need to run the app.py with a command in the terminal as (streamlit run app.py)
- it will localy host the file in your default web browser

And voila! your webapp will be runnning, feel free to add documents and chat with the agent.
Note : The models have token limit so the defualt slected model(LLama 3.1 turbo) should be able to handle only one small paper as it has a token limit
       of 8143 which is around 6000 words with prompt word count in mind.
       You can use the more powerful model (Deepseek v3) which has a token limit of 128000 and can easly handle multiple files.
