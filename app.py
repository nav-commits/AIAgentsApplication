from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

# Initial message
messages = [
    {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
]

# Print initial message
print(messages[0]["content"])

# Get user input
prompt = input("You: ")

# Append user message
messages.append({"role": "user", "content": prompt})

# Initialize LLM and tools
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
tools = [search, arxiv, wiki]

# Initialize search agent
search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

# Get response from agent
response = search_agent.run(messages)

# Append and print assistant response
messages.append({'role': 'assistant', "content": response})
print(messages)
print("Assistant:", response)
