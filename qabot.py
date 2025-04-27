from langchain.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ctransformers import CTransformers
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import LLMChain


persist_directory = 'docs/chroma/'

# Model embedding
embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-bf16.gguf")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# question = "What are major topics for this class?"
# docs = vectordb.similarity_search(question, k=3)


# Model 

llm = CTransformers(
        model = "models/vinallama-7b-chat_q5_0.gguf",
        model_type = "llama",
        config={"context_length": 2048},
        max_new_tokens = 1024,
        temperture  = 0.01)


# Build prompt
template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
Hello world!<|im_end|>
<|im_start|>assistant 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type = "stuff",
    retriever=vectordb.as_retriever(search_kwargs = {"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
question = "Is probability a class topic?"


resp = qa_chain.invoke({'query':question})
    
    
    