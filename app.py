import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# Sua chave de API da OpenAI
openai_api_key = "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"

# Defina a função add_vertical_space


def add_vertical_space(lines):
    for _ in range(lines):
        st.write("")  # Escreva uma linha em branco


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 Peter PDF File')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')

    # Chame a função para adicionar 5 linhas em branco
    add_vertical_space(5)

    st.write('by Pedro Marcelo')


def main():
    st.header("Wizard PDF")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # embeddings

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(
                api_key=openai_api_key, model="text-embedding-ada-002")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_area("Ask questions about your PDF file:")

        # Coloque o botão no canto
        st.write("")  # Adicione uma linha em branco para separar
        if st.button("Enviar Pergunta"):
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)

                # Exibir a resposta dividida em partes menores
                if isinstance(response, dict):
                    response_text = response.get('answer', 'No answer found.')
                else:
                    response_text = response
                max_chars_per_part = 2000
                response_parts = [response_text[i:i+max_chars_per_part]
                                  for i in range(0, len(response_text), max_chars_per_part)]
                for i, part in enumerate(response_parts):
                    st.write(part)


if __name__ == '__main__':
    main()
