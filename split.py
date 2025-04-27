from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


# Split for PDF file
loader = PyPDFLoader("./data/machinelearning-lecture01.pdf")
docs1 = loader.load()

### Simple split ###

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1000,
#     chunk_overlap=150,
#     length_function=len
# )

# document1 = text_splitter.split_documents(docs1)



# For embed split
text_splitter1 = RecursiveCharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

documents = text_splitter1.split_documents(docs1)




# CONTEXT AWARE SPLITTING

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.document_loaders import NotionDirectoryLoader

loader = NotionDirectoryLoader("File")
docs = loader.load()
txt = ' '.join([d.page_content for d in docs])

def markdown_header_text_split(docmask, headers):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(docsmask)
    return md_header_splits
docsmask = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Thuy\n\n Hi this is Clown\n\n \
### Section \n\n \
Hi this is Thuc \n\n 
## Chapter 2\n\n \
Hi this is Lemon"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
# For