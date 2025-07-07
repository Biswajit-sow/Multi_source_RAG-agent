# %%
import streamlit as st
import os
import requests
from pathlib import Path


#create a wrapper based on wikipidea and arxiv



#1st retriever tool-WIKIPEDIA



from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper



#using webbaseloader to load any data from any website ---3rd Retrievr tool

from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

# --- STREAMLIT UI SETUP ---
#Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"  # Enable tracing
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")  # Langsmith API key
os.environ["LANGCHAIN_PROJECT"] = "MULTI SOURCE RAG AGENT" # Give your project a descriptive name in langsmith


# Toggle Dark/White Mode using session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Embed CSS directly in the app
def apply_styles(dark_mode):
    st.markdown(f"""
        <style>
            body {{
                background-color: {'#262730' if dark_mode else '#FFFFFF'};
                color: {'#FFFFFF' if dark_mode else '#000000'};
            }}

            .stApp {{
                color: {'#FFFFFF' if dark_mode else '#000000'};
            }}

            .sidebar .sidebar-content {{
                background-color: {'#383A47' if dark_mode else '#F0F2F6'};
                color: {'#FFFFFF' if dark_mode else '#000000'};
            }}

            .stTextInput > div > div > input {{
                color: {'#FFFFFF' if dark_mode else '#000000'};
                background-color: {'#383A47' if dark_mode else '#FFFFFF'};
                border: 1px solid {'#555' if dark_mode else '#ccc'};
            }}
            .stTextArea > div > div > textarea {{
                color: {'#FFFFFF' if dark_mode else '#000000'};
                background-color: {'#383A47' if dark_mode else '#FFFFFF'};
                border: 1px solid {'#555' if dark_mode else '#ccc'};
            }}

            .stButton>button {{
                color: #FFFFFF;
                background-color: #4CAF50; /* Example button color */
                border: none;
            }}
            .stButton>button:hover {{
                background-color: #367C39;
            }}

             div.block-container {{
                  background-color: {"#8DA3C5" if not dark_mode else '#262730' };
             }}

             .streamlit-expanderHeader {{
                background-color: {'#D3D3D3' if not dark_mode else '#555'}; /* Slightly darker than F0F0F0 */
                color: {'#000000' if not dark_mode else '#FFFFFF'};
                padding: 0.5em;
                border-radius: 0.25em;
            }}
            /* Style the sidebar header */
            .sidebar .sidebar-header {{
              color: {'#000000' if not dark_mode else '#FFFFFF'};
            }}

           /* Target Streamlit elements directly */
            .css-1adrfps {{ /* This is the main content area */
                color: {'#000000' if not dark_mode else '#FFFFFF'} !important;
                background-color: {'#FFFFFF' if not dark_mode else '#262730'} !important;
            }}

            .css-12oz5g7 {{ /* Target sidebar elements more specifically */
                color: {'#000000' if not dark_mode else '#FFFFFF'} !important;
                background-color: {'#F0F2F6' if not dark_mode else '#383A47'} !important;
            }}
            /* Target sidebar elements even more specifically */
            .css-12oz5g7 .streamlit {{
                color: {'#000000' if not dark_mode else '#FFFFFF'} !important;
            }}



        </style>
    """, unsafe_allow_html=True)


apply_styles(st.session_state.dark_mode) # Apply initial style



# Sidebar toggle
st.sidebar.button("Toggle Dark Mode", on_click=toggle_dark_mode)

# Re-apply styles when the button is clicked
apply_styles(st.session_state.dark_mode)



st.title("📚 Multi-Source  Gemini RAG Agent")
st.info("Ask me questions about LangSmith, Wikipedia, or Arxiv papers!")

# --- ENVIRONMENT AND API KEY SETUP ---
# Load environment variables
load_dotenv()

# Check for API key and handle it securely
if 'GOOGLE_API_KEY' not in os.environ:
    try:
        os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY")
    except Exception:
        st.error("GEMINI_API_KEY not found in .env file. Please add it.")
        st.stop()

if not os.environ.get('GOOGLE_API_KEY'):
    st.error("Google API Key is not set. Please add it to your .env file or environment variables.")
    st.stop()


# --- AGENT AND TOOLS SETUP (CACHED FOR EFFICIENCY) ---
# Use st.cache_resource to initialize the agent and tools only once.
@st.cache_resource
def initialize_agent():
    st.write("Initializing agent and tools... (This runs only once)")

    # %%
    # --- 1st retriever tool-WIKIPEDIA ---
    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)#top_k_result define how many result you want and doc _content_chars_max define how many character you want to show in the summary
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

    st.write(f"✅ Loaded Tool: {wiki.name}")

    # --- 2nd Retriever tool: WebBaseLoader for LangSmith ---
    # Load webpage
    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()

    # Split into manageable chunks
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    # Use Gemini-2.0-Flash with your Gemini API Key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # Updated to 1.5-flash for better performance

    # Correct embedding model for Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create vector DB and retriever
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever()

    #create a retriever tool
    #Create a tool to do retrieval of documents.
    from langchain.tools.retriever import create_retriever_tool
    #it is basically creating a tool to do the search for that particular page
    retriever_tool=create_retriever_tool(retriever,"langsmith_search",
                          "search  for information about Langsmith.For any question about Langsmith you must use this tools")#retriever=retriever name,langsmith_search help to identify the tool name

    st.write(f"✅ Loaded Tool: {retriever_tool.name}")


    #3rd Retriever tool:ARXIV platform
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)#top_k_result define how many result you want and doc _content_chars_max define how many character you want to show in the summary
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)# create a arxiv wrrapper

    st.write(f"✅ Loaded Tool: {arxiv.name}")

    # %%
    tools=[wiki,arxiv,retriever_tool]


   # Step 2 :Query from the specific tools


    #There are two way to creating a prompt one is chat prompt  template and one is langchain hub- this is a generic prompt which is crate by many other user of langchain and also crate by langchain team.

    #get the prompt to use -you can modify this to suit your needs
    prompt= hub.pull("hwchase17/openai-functions-agent")# 1st one is username that is prsent in the langchainhub  and other one is name of the function that you want to use


    """
    Gemini_Tool Agent
    """
    # Agent setup using LCEL Tool Calling
    agent=create_openai_tools_agent(llm,tools,prompt)

    # %%
    #to use this agent we have to use agent executer
    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)#agentname ,tools name ,verbose help to get the details whenever I am getting a respone

    st.success("Agent and tools initialized successfully!")
    return agent_executor

# Initialize the agent by calling the cached function
agent_executor = initialize_agent()


# --- INTERACTIVE CHAT INTERFACE ---
# Get user input from a text box
user_input = st.text_input("Enter your query:", placeholder="e.g., Tell me about LangSmith or what is Data Science?")

if user_input:
    # Display a spinner while processing the request
    with st.spinner("Agent is thinking..."):
        try:
            # Create a container for the agent's thought process
            with st.expander("Agent's Thought Process"):
                # Use a custom handler to stream thoughts to the UI
                # NOTE: For complex streaming, StreamlitCallbackHandler is great.
                # Here, we will capture the output from invoke which is simpler.
                # The verbose=True in AgentExecutor prints to console, we capture the final output here.
                st.write("Invoking agent with your query...")

                # Execution with Gemini
                result = agent_executor.invoke({"input": user_input})

                # The 'verbose=True' output will appear in your console/terminal.
                # We display the final answer below.
                # A more advanced implementation could redirect stdout to display here.
                st.write("Agent finished processing.")

            # Display the final result
            st.subheader("Answer:")
            st.write(result.get("output", "No output found."))

        except requests.exceptions.RequestException as e:
            st.error(f"Network Error: {e}")
        except (KeyError, ValueError) as e:
            st.error(f"Error parsing response: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Example questions to guide the user
st.sidebar.title("Example Questions")
st.sidebar.markdown("""
- Tell me about LangSmith.
- What is Data Science?
- What's the paper 1605.08386 about?
- Who is the president of the United States?
""")