# LangGraph Agentic Chatbot

Deployed on HuggigFace Spaces. Here is the link to the live site: [https://huggingface.co/spaces/akileshjayakumar/LangGraph-Agentic-Chatbot](https://huggingface.co/spaces/akileshjayakumar/LangGraph-Agentic-Chatbot)

This repository contains a LangGraph-based chatbot that utilizes Gradio for interface rendering and integrates OpenAI's language model capabilities and custom tools for enhanced functionality. This project allowed me to learn and explore the capabilities of LangGraph and Agnetic workflows.

## Tech Stack


- **Frontend:**

  - **[Gradio](https://gradio.app/docs)**

- **Backend:**

  - **[LangGraph](https://langgraph.dev/)**
  - **[LangChain](https://python.langchain.com/en/latest/)**
  - **[Python](https://www.python.org/)**

- **APIs:**

  - **[OpenAI API](https://platform.openai.com/docs)**
  - **[Tavily Search API](https://tavilyapi.com/docs)**

- **Version Control:**
  - **[Git](https://git-scm.com/doc)**

## Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/langgraph-agentic-chatbot.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd langgraph-agentic-chatbot
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**

   Create a `.env` file in the root directory and add your OpenAI API key and LangChain API key to trace the chatbot's interactions.

   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.example.langchain.com"
   LANGCHAIN_API_KEY="your_langchain_api_key"
   LANGCHAIN_PROJECT="your_langchain_project"
   ```

   ```
   OPENAI_API_KEY="your_openai_api_key"
   TAVILY_API_KEY="your_tavily_api_key"
   ```

5. **Run the Gradio application:**

   ```bash
   gradio agent/main.py
   ```

6. **Access the chatbot:**

   Open `http://localhost:7860` in your web browser to interact with the chatbot.
