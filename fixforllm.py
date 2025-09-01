from langchain_community.document_loaders import PyMuPDFLoader
import os

folder_name = "data"
files = os.listdir(folder_name)

docs = []
for file in files:
    if not file.endswith(".pdf"):
        continue
    loader = PyMuPDFLoader(f"{folder_name}/{file}")
    pages = loader.load()
    docs.extend(pages)

    from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=30,
    separator="\n",
)

splitted_pages = text_splitter.split_documents(docs)

from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

from langchain.vectorstores import Chroma

if os.path.isdir(".healthX_db"):
    db = Chroma(persist_directory=".healthX_db", embedding_function=embeddings)
else:
    db = Chroma.from_documents(splitted_pages, embedding=embeddings, persist_directory=".healthX_db")

retriever = db.as_retriever()

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

query = "HealthXの無料プランについて教えてください。"

result = chain.run(query)
print(result)