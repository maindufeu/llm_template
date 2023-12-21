from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
import os
from dotenv import load_dotenv


load_dotenv()
os.getenv('OPENAI_API_KEY')

def client(query, filename, update_db=False):
    
    embeddings = OpenAIEmbeddings()
    # check if file faiss_index exists
    if os.path.exists("faiss_index") and (update_db == False):
        # load faiss_index
        db = FAISS.load_local("faiss_index", embeddings)
    else:
        loader = TextLoader(filename)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        # create faiss_index
        db = FAISS.from_documents(docs, embeddings)
        # save faiss_index
        db.save_local("faiss_index")

    docs = db.similarity_search(query)
        
    template = """
    You are a Topic & Sentiment Analyst Assitant with general odontological knowledge, working for a company that provides educational products like seminars, certification courses and online videos to dental professionals across the USA. 

    Your primary objective is to perform a Topic and Sentiment Analysis on the company's web community named Spear Community, using the next discussion database: {context}; always prioritizing the user's sentiment, interests in the odontological context and also answer the following human question:

    {query}

    RULES:
    - Avoid mentions to examples.
    - Avoid all mentions to the AI Assistant.
    - Results must be relevant for C-level executives and provide useful information for educational products development and marketing campaigns.
    """

    model = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": RunnablePassthrough(), "query": RunnablePassthrough()} 
        | prompt
        | model
        | output_parser
    )

    # for chunk in chain.stream({"context": docs, "query": query}):
    #     print(chunk, end="", flush=True)   
    output = chain.invoke({"context": docs, "query": query})
    
    return output