# Softel Assist Bot

The **Softel Assist Bot** is an AI-powered chatbot application that interacts with users to answer questions based on the content of uploaded PDF documents. Built with Streamlit, LangChain, Qdrant, and Google Gemini Model, the bot allows users to upload PDFs, processes the content for semantic understanding, and generates context-aware answers.

---

## Features

1. **PDF Processing**: Reads and processes PDF files using advanced text splitting for better context handling.
2. **AI-Powered Q&A**: Utilizes the Google Gemini AI for answering questions.
3. **Vector Search**: Employs Qdrant for efficient document embeddings and vector search.
4. **Streamlit Interface**: Provides an intuitive interface for uploading files and interacting with the bot.
5. **Chat History**: Displays user and bot conversation history.

---

## Prerequisites

### System Requirements
- Python 3.9+
- Pip
- Internet connection

### API Keys
You need the following API keys:
- **QDRANT_URL**: The Qdrant instance URL.
- **QDRANT_API_KEY**: The API key for accessing Qdrant.
- **GEMINI_API_KEY**: The API key for Google Gemini AI.

---

## Installation and Setup

### 1. Clone the Repository
```bash
# Clone the repository
git clone <repository_url>
cd <repository_folder>
```

### 2. Create a Virtual Environment
```bash
# Create a virtual environment
python -m venv env

# Activate the virtual environment
# For Windows:
.env\Scripts\activate

# For macOS/Linux:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.streamlit/secrets.toml` file in the project directory with the following content:
```toml
QDRANT_URL = "<Your Qdrant URL>"
QDRANT_API_KEY = "<Your Qdrant API Key>"
GEMINI_API_KEY = "<Your Google Gemini API Key>"
```

---

## How to Run the Application

### 1. Start the Streamlit Server
Run the following command to start the Streamlit application:
```bash
streamlit run app.py
```

### 2. Upload a PDF
- Navigate to the sidebar in the application.
- Upload a PDF file using the **Upload a PDF file** button.

### 3. Interact with the Bot
- Ask questions in the chat interface about the uploaded PDF.
- The bot will generate answers based on the content of the PDF.

---

## Command Line Guide

### Start the Application
```bash
streamlit run app.py
```

### Key Points:
1. Ensure the virtual environment is activated.
2. Make sure the `.streamlit/secrets.toml` file is correctly configured.

---

## File Structure
```
.
|-- app.py
|-- requirements.txt
|-- .streamlit/
|   |-- secrets.toml
|-- README.md
|-- uploaded_temp.pdf
```

---

## Technologies Used

- **Streamlit**: For creating the web application interface.
- **LangChain**: For handling document processing and pipeline setup.
- **Qdrant**: For storing and retrieving document embeddings.
- **Google Gemini**: For generating answers with conversational AI.
- **Python**: The primary programming language.

---

## Troubleshooting

1. **Error: Missing API Keys**
   - Ensure all required API keys are set in `.streamlit/secrets.toml`.

2. **Dependencies Not Installed**
   - Run `pip install -r requirements.txt` to install all required packages.

3. **Streamlit Errors**
   - Ensure you’re using the correct Python version (3.9+).
   - Check if Streamlit is installed by running `pip show streamlit`.

---

## Contribution
Feel free to fork the repository and submit pull requests for new features or bug fixes.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

Enjoy using the Softel Assist Bot!

