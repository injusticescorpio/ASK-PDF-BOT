import streamlit as st
from dotenv import load_dotenv
import os
from os.path import join, dirname
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

#for tracking token usage
from langchain.callbacks import get_openai_callback

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

def main():
    # print(os.environ.get("OPENAI_API_KEY"))
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask you PDF ❓")
    #upload pdf
    pdf=st.file_uploader("Upload your PDF",type="pdf")
    #extract pdf files
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #divide text into chunks 
        text_splitter=RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        chunks=text_splitter.split_text(text)
        print(len(chunks))
        #create embeddin
        embeddings=OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks,embeddings)
        
       
        #show user input
        user_question=st.text_input("Ask a question about your PDF...")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
  
            
 
        
if __name__ == "__main__":
    main()