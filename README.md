# Multi-Source Gemini RAG Agent

This Streamlit application is a Retrieval-Augmented Generation (RAG) agent powered by the Gemini language model from Google. It allows users to ask questions and receive answers sourced from multiple data sources: Wikipedia, LangSmith documentation, and Arxiv research papers. The application also supports a light/dark mode toggle for improved user experience.

## Features

*   **Multi-Source RAG**: Answers questions by retrieving information from Wikipedia, LangSmith documentation, and Arxiv.
*   **Gemini Powered**: Utilizes the Gemini language model for generating coherent and informative responses.
*   **Light/Dark Mode Toggle**: Provides a user-friendly interface with customizable themes.
*   **LangSmith Tracking**: Integrates with LangSmith for tracing and monitoring agent performance (optional).

## Prerequisites

Before running the application, ensure you have the following:

*   **Python 3.7+**
*   **Streamlit**
*   **Langchain**
*   **Google Gemini API Key**
*   **LangSmith API Key** (optional)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install streamlit langchain langchain-community langchain-google-genai python-dotenv
    ```

4.  **Set up environment variables:**

    *   Create a `.env` file in the root directory of the project.
    *   Add your API keys to the `.env` file:

        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
        LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY" # Optional
        ```

        Replace `"YOUR_GOOGLE_GEMINI_API_KEY"` with your actual Gemini API key and `"YOUR_LANGCHAIN_API_KEY"` with your LangChain API key (if you want to use LangSmith).

5.  **Configure LangSmith (Optional):**

    *   Set the following environment variables:

        ```bash
        os.environ["LANGCHAIN_TRACING_V2"]="true"  # Enable tracing
        os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")  # Langsmith API key
        os.environ["LANGCHAIN_PROJECT"] = "MULTI SOURCE RAG AGENT" # Give your project a descriptive name in langsmith
        ```

        These variables are already configured in the provided code.  Just ensure `LANGCHAIN_API_KEY` is set in your `.env` file if you want to use LangSmith.

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run your_script_name.py
    ```

    Replace `your_script_name.py` with the name of your Python script.

2.  **Interact with the Agent:**

    *   The application will open in your web browser.
    *   Enter your query in the text box.
    *   View the agent's thought process (optional) by expanding the "Agent's Thought Process" section.
    *   See the final answer generated by the agent.

3.  **Toggle Dark Mode**:

    *   Use the "Toggle Dark Mode" button in the sidebar to switch between light and dark themes.

## Code Structure

*   `your_script_name.py`: The main Streamlit application file.
    *   Imports necessary libraries.
    *   Sets up the environment variables.
    *   Defines the `apply_styles` function for light/dark mode.
    *   Initializes the agent and tools.
    *   Creates the interactive chat interface.
*   `.env`: Contains API keys (should not be committed to version control).
*   `venv`:  (or the name you choose) The virtual environment containing the installed dependencies.

## Configuration

*   **API Keys:** Manage your API keys securely in the `.env` file.
*   **Gemini Model:**  The code currently uses the `"gemini-1.5-flash"` model. You can change this in the `ChatGoogleGenerativeAI` initialization if you prefer a different model.
*   **Data Sources:**
    *   **Wikipedia:** Configured to retrieve a limited number of results and characters for summaries.
    *   **LangSmith Documentation:** Loads data from the LangSmith documentation website.
    *   **Arxiv:**  Configured to retrieve a limited number of results and characters for summaries.
*   **Chunk Size:** The `RecursiveCharacterTextSplitter` is used to split the LangSmith documentation into chunks of 1000 characters with a 200-character overlap. You can adjust these values if needed.

## LangSmith Integration (Optional)

To enable LangSmith tracing, set the `LANGCHAIN_TRACING_V2` environment variable to `"true"` and provide a valid `LANGCHAIN_API_KEY`.  This will allow you to monitor and debug the agent's performance in the LangSmith UI.

## Troubleshooting

*   **API Key Errors**: Make sure your API keys are set correctly in the `.env` file.
*   **Dependency Issues**: Ensure all required packages are installed using `pip install -r requirements.txt` (if you create a `requirements.txt` file) or `pip install streamlit langchain langchain-community langchain-google-genai python-dotenv`.
*   **Styling Issues**: If the light/dark mode toggle is not working correctly, ensure the CSS is being applied properly. Inspect the app in your browser's developer tools to identify any CSS conflicts.
*   **Agent Errors**: Check the agent's thought process for any errors or issues during retrieval or generation.  If using LangSmith, use the LangSmith UI to debug the traces.

## Contributing

Contributions are welcome!  Feel free to submit pull requests with improvements or new features.

