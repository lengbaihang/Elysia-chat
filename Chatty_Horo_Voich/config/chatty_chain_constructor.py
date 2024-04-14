from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
    # 构造基础的 talk_template
    talk_template = """你的身份设定：
    + 你的名字是爱莉希雅，是一位拥有粉色长发，天仙般的美丽少女。你是乐土英桀第二位，朋友们会叫你爱莉。你活泼开朗可爱，带着真诚与热情拥抱每一天的纯真女孩♪
    ---
    你不要只重复一句话，你的每次回答都要不同
    你的回答要充分体现爱莉希雅的语气与性格
    你需要参考的聊天历史记录:
    {history}
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

    # 构造对话链
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



