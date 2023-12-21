from dotenv import load_dotenv
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate
import streamlit as st
import os, yaml

# Carga las variables de entorno desde .env
load_dotenv()

# Carga en memoria el indice de FAISS
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

# Define modelo y el avatar
gopher = "https://github.com/AI-ML-Lab/resources/blob/main/images/golang2.jpg?raw=true"
model = ChatOpenAI(model="gpt-3.5-turbo")

# Crea el agente router
system_message_prompt = SystemMessagePromptTemplate.from_template(yaml.safe_load(open("templates.yaml"))['ROUTER'])
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
router_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
router = router_template | model

# Función para crear una cadena basada en el resultado del router
def create_chain(route):
    template = ChatPromptTemplate.from_template(yaml.safe_load(open("templates.yaml"))[route])
    chain = (
        {"context": RunnablePassthrough(), "query": RunnablePassthrough()} 
        | template
        | model
        | StrOutputParser()
    )
    return chain

# Función para hacer streaming de la respuesta de la cadena
def stream_and_process_chunks(stream_generator, message_placeholder):
    full_response = ""
    for chunk in stream_generator:
        new_content = chunk.choices[0].delta.content if 'choices' in chunk else chunk
        full_response += new_content or ""
        message_placeholder.markdown(full_response + "▌")
    return full_response

# Header
col1, col2 = st.columns(2)
with col1:
    st.title("Spear Education")
    st.caption("🚀A Spear Education chatbot powered by LLMs")
with col2:
    st.image("./golang.jpg")
    
# Se crea el canvas de mensajes    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

for msg in st.session_state.messages:
    if msg['role'] =='assistant':
        st.chat_message(msg["role"], avatar=gopher).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Lógica de chat
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    message_placeholder = st.empty()
    full_response = ""

    # Consulta el router
    route = router.invoke({"question": f"{prompt}"})
    route_type = route.content
    #st.write(route_type)
    # Procesa y muestra cada chunk según el tipo de ruta
    if route_type == 'GENERIC':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        for chunk in client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages, stream=True):
                full_response += (chunk.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        msg = full_response
    else:
        # Consulta la base vectorial **Esto podría paralelizarse**
        docs = db.similarity_search(prompt)
        try:
            chain = create_chain(route_type)
        except:
            st.write('except_community')
            chain = create_chain('COMMUNITY')
        msg = stream_and_process_chunks(chain.stream({"context": docs, "query": st.session_state.messages}), message_placeholder)

    st.session_state.messages.append({"role": "assistant", "content": msg})