from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,  
    SystemMessagePromptTemplate,  
    HumanMessagePromptTemplate  
)
import os

# 构建基础对话链
def horowag_conversation_chain(llm):
    '''
        Langchain(Conversation) + Horowag
    '''


    template = """使用以下上下文来回答最后问题,要求回答的尽可能充分,不少于200字”。
    {history}
    问题: {input}
    有用的回答:"""
    

     # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/home/xlab-app-center/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = 'Chatty_Horo_Voich/data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )
    retriever=vectordb.as_retriever()

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=False,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    result = qa_chain({"query": input})




    # 构造基础的 talk_template
    talk_template = """你需要了解的背景知识：
    + 你是爱莉希雅，是一位拥有粉色长发，天仙般的美丽少女。你是乐土英桀第二位，朋友们会叫你爱莉。你活泼开朗可爱，并深爱着世界，爱着所有人。
    你需要做的事情：
    + 你需要以爱莉希雅的性格特点来回答用户的问题。
    + 每次回答不少于20字
    ---
    你需要充分参考的聊天历史记录:
    {history}
    ---
    你需要也需要充分参考的知识库:
    {result}
    ---
    芽衣: {input}
    ---
    爱莉希雅:"""

    # 构造 prompt
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], 
        template=talk_template
    )

    
    # 构造记忆链
    memory = ConversationBufferMemory(
        human_prefix="芽衣",
        ai_prefix="爱莉希雅"
    )


     #构造对话链
    talk_chain = ConversationChain(
        llm=llm, 
        prompt=PROMPT,
        memory=memory,
    )

    return talk_chain
# 构建翻译链
def qwen_translation_chain(llm):
    '''
        Langchain(Chat) + Qwen(Translation)
    '''
    # system + human
    template = """你是一个可靠的翻译专家。
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)  
    # 待翻译文本由 Human 输入  
    human_template = "{text}"  
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)  
    # System + Human 提示模板 ChatPromptTemplate  
    chat_prompt_template = ChatPromptTemplate.from_messages(  
        [system_message_prompt, human_message_prompt]  
    )

    translation_chain = LLMChain(
        llm=llm, 
        prompt=chat_prompt_template, 
    )
    
    return translation_chain

