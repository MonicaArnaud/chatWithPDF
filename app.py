# Import necessary modlues 

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI 
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, 
)
import os
import io 
import chainlit as cl 
import PyPDF2 
from io import BytesIO 

from dotenv import load_dotenv 

# load_dotenv() 

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Text splitter and system template
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """ Use the following pieces of context to answer the users questions.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:
'''
The answer is foo
SOURCES: xyz
'''
Begin!
---------

{summaries} """

messages = {
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")

}
prompt = ChatPromptTemplate.from_template(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start 
async def on_chat_start():
    
    # Sending an image with the local file path
    elements = [
        cl.Image(name="Robot1", display="inline", path="userProfil.jpg")
    ]
    await cl.Message(content="你好，欢迎来到PDF聊天室"， elements=elements).send()
    files = None
    
    # wait for the user to upload the file
    while file is None:
        file = await cl.AskFileMessage(
            content="请上传文件",
            accept = ["application/pdf"],
            max_size_mb=20,
            timeout=180).send()
        
    file = files[0]
    
    msg = cl.Message(content="正在处理文件'{file.name}'请稍等")
    await msg.send() 
    
    # Read the pdf file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream) 
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extractText()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)
    
    
    # Create metadata for each chunk 
    metadatas  =  [{"Source": f"{i}-pl"} for i in range(len(texts))]
    
    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadata = metadatas
    )

    # Create a chain that uses the Chroma vector store 
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type = "stuff",
        retriever = docsearch.as_retriever()
    )
    
    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    
    # Let the user know that the system is ready
    msg.content = f"'{file.name}处理完成，你现在可以问问题了！"
    await msg.update()
    
    cl.user_session.set("chain", chain)

@cl.on_message 
async def main(message:str):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangChainCallbackHandler(
        stream_final_answer = True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True 
    res = await chain.acall(message, callbacks=[cb]) 
    
    answer = res['answer']
    sources = res["sources"].strip()
    source_elements = []
    
    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")
    
    
    if sources: 
        found_sources = []
        
        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue 
            
            text = texts[index] 
            found_sources.append(source_name) 
            source_elements.append(c.Text(content=text, name=source_name))
        
        if found_sources:
            answer += f"\n来源：{','.join(found_sources)}"
        else:
            answer += f"\n没有找到任何来源"
    
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update() 
    else:
        await cl.Message(content=answer, elements = source_elements).send()
        
        



    
    



