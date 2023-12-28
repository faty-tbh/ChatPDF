#installation PyPDF2, langchain, python-dotenv steamlit-chat
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit import message
st.set_page_config(page_title="ChatPDF",page_icon=":globe2:",layout="wide")

#--GET PDF, EXTRACT TEXT, DIVIDE CHUNKS--#
def get_extract_chunks(pdf_docs): 
    #pdf_docs is an array of uploaded pdfs
    content = ""
    #Strat content extraction
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content += page.extract_text()
    #En Extraction

    #Start Chunks Breakdown 
    splitter= CharacterTextSplitter(
        separator="\n",
        chunk_size =1400,
        chunk_overlap=320,
        lenght_function=len
    )
    chunks=splitter.split_text(content)

    return chunks
def create_vectorstore(chunks):
    current_embedding= OpenAIEmbeddings()
    vectorestore= faiss.from_texts(texts=chunks,
    embedding= current_embedding)
    return vectorestore

def get_conersation_chain(vectorestore):
    current_llm=ChatOpenAI()
    current_memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm= current_llm,
        retriever=vectorestore.as_retriever(),
        memory=current_memory
    )
    return conversation_chain

def get_user_input(user_input):
    if not st.session_state.pdf_processed:
        st.info('Please upload PDF files')
        return
    answer=st.session_state.conversation({'question':user_input})
    st.session_state.chat_history=answer['chat_history']
    for idx, value in enumerate(st.session_state.chat_history):
        if idx % 2 ==0:
            message(value, is_user=True, key=str(idx)+'_user')
        else:
            message(value, key=str(idx))
    

def main():
    load_dotenv()
    #---UPLOAD PDF---#

    st.header("Hey, chat locally here ! :flag-cd:")
    if 'converstion' not in st.session_state:
        st.session_state.conversation =None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history =None  

    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed=False 

    user_input= st.text_input("Please Enter your question..")    
    if user_input:
        get_user_input(user_input)  
    #First method:
    # st.sidebar.subheader("Please apload PDFs")
    # st.sidebar.file_uploader("Upload files", type="file",accept_multiple_files=True)
    #Second method:
    with st.sidebar:
        st.subheader("Please apload PDFs")
        pdf_docs=st.file_uploader("Upload files", type="file",accept_multiple_files=True)
        button=st.button("Extraction")
        #okay=False

        if button:
           with st.spinner("Processing..."):
                #Get chunks
                chunks= get_extract_chunks(pdf_docs)
                #okay=True
                # Create vectorstore
                vectorestore= create_vectorstore(chunks)
                if st.session_state.conversation: get_conersation_chain(vectorestore)
    # if okay:
    #     for chunk in chunks :
    #         st.success(chunk)



#execution des fonctions qui sont definis dans la fonction main, condition __name__ =='__main__'
if __name__ =='__main__':
    main()