import os
import bcrypt
import streamlit as st
import sqlite3
import pyttsx3
import threading
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from collections import defaultdict
import hashlib
import time
import pandas as pd
from textblob import TextBlob
import faiss
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Get the OpenAI API key from the environment
openai_api_key = os.getenv('OPENAI_API_KEY')

# Ensure the API key is available
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Caching mechanism
cache = {}

def get_hash_for_url(url):
    return hashlib.md5(url.encode()).hexdigest()

def get_vectorstore_from_url(url):
    url_hash = get_hash_for_url(url)
    if url_hash in cache:
        return cache[url_hash]
    
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    cache[url_hash] = vector_store

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(api_key=openai_api_key)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI(api_key=openai_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store, user_tone, response_length):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    sentiment, subjectivity = analyze_sentiment(user_input)
    if sentiment < 0:
        user_tone = "empathetic"
    
    personalized_input = f"Respond in a {user_tone} tone with a {response_length} response length. {user_input}"
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": personalized_input
    })
    
    return response['answer'], sentiment, subjectivity

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def save_chat_history_to_db(username):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    with conn:
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                    (username TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        for message in st.session_state.chat_history:
            c.execute("INSERT INTO chat_history (username, role, content) VALUES (?, ?, ?)",
                      (username, "AI" if isinstance(message, AIMessage) else "Human", message.content))
    conn.close()

def load_chat_history_from_db(username):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    st.session_state.chat_history = []
    c.execute("SELECT role, content FROM chat_history WHERE username = ? ORDER BY timestamp", (username,))
    for row in c.fetchall():
        if row[0] == "AI":
            st.session_state.chat_history.append(AIMessage(content=row[1]))
        else:
            st.session_state.chat_history.append(HumanMessage(content=row[1]))
    conn.close()

def visualize_knowledge_graph():
    G = nx.Graph()
    
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            G.add_node(message.content, color='blue')
        else:
            G.add_node(message.content, color='red')
    
    pos = nx.spring_layout(G)  # use spring layout for better spacing
    colors = [node[1]['color'] for node in G.nodes(data=True)]
    
    plt.figure(figsize=(10, 8))  # increase figure size
    nx.draw(G, pos, node_color=colors, with_labels=True, font_size=10, node_size=3000, font_weight='bold')
    st.pyplot(plt.gcf())

def summarize_chat_history():
    llm = ChatOpenAI(api_key=openai_api_key)
    conversation_history = "\n".join([msg.content for msg in st.session_state.chat_history if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)])
    summary_prompt = f"Summarize the following conversation:\n\n{conversation_history}"
    response = llm.invoke(summary_prompt)
    return response

def speak_text(text):
    def run_speak():
        engine.say(text)
        engine.runAndWait()

    speak_thread = threading.Thread(target=run_speak)
    speak_thread.start()

def provide_general_ai_voice_instruction():
    instructions = """
    Welcome to the Chat with Websites application! Here are some tips to get you started:
    
    1. In the sidebar, you can enter a website URL. The bot will use this URL to fetch and process information.
    2. You can choose the tone of the bot: Friendly, Professional, Humorous, Informative, or Empathetic.
    3. You can also choose the response length: short, medium, or long.
    4. If you have a specific prompt or question for the bot, you can enter it in the Custom Prompt area.
    5. You can save your chat history by clicking the "Save Chat History" button.
    6. Load previous chat history with the "Load Chat History" button.
    7. Visualize the chat history as a knowledge graph with the "Visualize Knowledge Graph" button.
    8. Summarize the entire chat session using the "Summarize Chat History" button.
    
    Start chatting by typing your message in the input box below and press enter. The bot will respond and read out the response aloud. Enjoy your experience!
    """
    speak_text(instructions)

# Passwords for users
names = ["ADHI JAG", "ADHI WICK"]
usernames = ["adhijag", "adhiwick"]
passwords = ["123", "456"]

# Hash passwords using bcrypt
hashed_passwords = {username: bcrypt.hashpw(password.encode(), bcrypt.gensalt()) for username, password in zip(usernames, passwords)}

# Function to authenticate users
def authenticate_user(username, password):
    if username in usernames:
        hashed = hashed_passwords[username]
        return bcrypt.checkpw(password.encode(), hashed)
    return False

# Authentication form
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful")
            provide_general_ai_voice_instruction()
        else:
            st.error("Username or password is incorrect")
else:
    st.title("Chat with websites")
    
    # Initialize session state variables
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    
    # sidebar
    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")
        bot_tone = st.selectbox("Choose Bot Tone", ["Friendly", "Professional", "Humorous", "Informative", "Empathetic"])
        response_length = st.selectbox("Response Length", ["short", "medium", "long"])
        custom_prompt = st.text_area("Custom Prompt")
        if st.button("Save Chat History"):
            save_chat_history_to_db(st.session_state.username)
        if st.button("Load Chat History"):
            load_chat_history_from_db(st.session_state.username)
        if st.button("Visualize Knowledge Graph"):
            visualize_knowledge_graph()
        if st.button("Summarize Chat History"):
            summary = summarize_chat_history()
            st.session_state.chat_history.append(AIMessage(content=f"Summary: {summary}"))
    
    if website_url is None or website_url == "":
        st.info("Please enter a website URL")

    else:
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
        
        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            try:
                response, sentiment, subjectivity = get_response(user_query, st.session_state.vector_store, bot_tone, response_length)
                st.session_state.chat_history.append(HumanMessage(content=f"{user_query} (Sentiment: {sentiment}, Subjectivity: {subjectivity})"))
                st.session_state.chat_history.append(AIMessage(content=response))
                speak_text(response)  # Voice guidance
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
                    speak_text(message.content)  # Read AI message aloud
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    
        # User Analytics
        st.sidebar.subheader("User Analytics")
        st.sidebar.text(f"Total Messages: {len(st.session_state.chat_history)}")
        st.sidebar.text(f"Session Duration: {int(time.time() - st.session_state['start_time'])} seconds")









