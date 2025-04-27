
# # For PDF file 
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./data/machinelearning-lecture01.pdf")
docs1 = loader.load()


# For Youtobe
from langchain.document_loaders import YoutubeLoader

url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
loader = YoutubeLoader.from_youtube_url(url)
docs2 = loader.load()

# FOR URL 
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://truyenfull.vision/tien-nghich/chuong-480/")
docs3 = loader.load()






