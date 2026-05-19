import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace


## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

HF_TOKEN=os.environ.get("HF_TOKEN") 
DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

@st.cache_resource
def load_llm(huggingface_repo_id , HF_TOKEN):

    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        task = "conversational",
        max_new_tokens=512
    )

    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        huggingface_repo_id = "meta-llama/Llama-3.1-8B-Instruct"
        HF_TOKEN=os.environ.get("HF_TOKEN")  
        llm = load_llm(huggingface_repo_id=huggingface_repo_id, HF_TOKEN=HF_TOKEN)

        try :
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                                                return_source_documents=True,
                                                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)})

            response = qa_chain.invoke({"query": prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")
        # #TODO: Create a Groq API key and add it to .env file
        
        # try: 
        #     vectorstore=get_vectorstore()
        #     if vectorstore is None:
        #         st.error("Failed to load the vector store")

        #     qa_chain = RetrievalQA.from_chain_type(
        #         llm=ChatGroq(
        #             model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, fast Groq-hosted model
        #             temperature=0.0,
        #             groq_api_key=os.environ["GROQ_API_KEY"],
        #         ),
        #         chain_type="stuff",
        #         retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
        #         return_source_documents=True,
        #         chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        #     )

        #     response=qa_chain.invoke({'query':prompt})

        #     result=response["result"]
        #     source_documents=response["source_documents"]
        #     result_to_show=result+"\nSource Docs:\n"+str(source_documents)
        #     #response="Hi, I am MediBot!"
        #     st.chat_message('assistant').markdown(result_to_show)
        #     st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        # except Exception as e:
        #     st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()