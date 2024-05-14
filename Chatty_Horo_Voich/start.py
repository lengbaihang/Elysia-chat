from config.chatty_chain_constructor import horowag_conversation_chain
from config.chatty_model_rebuilder import  Horowag
from langchain.prompts import PromptTemplate
from openxlab.model import download
import subprocess
import gradio as gr
import sys
import os

# 预设变量
__file__ = " "
theme = 'ParityError/Anime'
# /root/.conda/envs/test/lib/python3.10/site-packages/pyopenjtalk/

# 构建编译环境
os.system("export CUDA_LAUNCH_BLOCKING=1")
os.system("pip install imageio==2.4.1")
os.system("pip install moviepy==1.0.1")
os.system("pip install pyopenjtalk==0.3.3")
os.system("tar -zxvf open_jtalk_dic_utf_8-1.11.tar.gz -C /usr/local/share/python/.pyenv/versions/3.9.16/lib/python3.9/site-packages/pyopenjtalk/")
print("声音编译环境构建完成...")

# 回归任务环境
os.chdir('Chatty_Horo_Voich/VITS-kit/monotonic_align')
os.system("mkdir monotonic_align")
os.system("python setup.py build_ext --inplace")

# 环境命令
os.chdir('/home/xlab-app-center/')

# 加载基础的语言模型 Horowag_7b

#os.system('git clone https://code.openxlab.org.cn/lengbaihang1/Elysia03.git  ./Horowag_7b')

base_path = 'Horowag_7b'
os.system(f'git clone https://code.openxlab.org.cn/lengbaihang1/Elysia04.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

#download(model_repo='lengbaihang1/Elysia03',     ######需要更改
 #        output='Horowag_7b')
print("Horowag_7b 下载完毕")

# 加载辅助的语言模型 Qwen1_5
#os.system('git clone https://code.openxlab.org.cn/lengbaihang1/Qwen1.5_4B_Auxiliary_AWQ.git  ./Qwen_Auxiliary_AWQ')

#base_path = 'Qwen_Auxiliary_AWQ'
#os.system(f'git clone https://code.openxlab.org.cn/lengbaihang1/Qwen1.5_4B_Auxiliary_AWQ.git {base_path}')
#os.system(f'cd {base_path} && git lfs pull')

#download(model_repo='lengbaihang1/Qwen1.5_4B_Auxiliary_AWQ',     #####需要更改
 #        output='Qwen_Auxiliary_AWQ')
#print("Qwen_Auxiliary_AWQ 下载完毕")

# 加载语音微淘模型 Speaker
#os.system('git clone https://code.openxlab.org.cn/lengbaihang1/Elysiavits4.git ./Elysiavits4')

base_path = 'Elysiavits'
os.system(f'git clone https://code.openxlab.org.cn/lengbaihang1/Elysiavits5.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

my_list = []

#download(model_repo='lengbaihang1/Elysiavits4',     #######需要更改
#         output='/home/xlab-app-center/')
print("Speaker_Tuning_Model 下载完毕")

# Qwen 模型初始化

# 构建翻译链
#qwen_translation_chain = qwen_translation_chain(Qwen_model)

# 定义音频构建函数
def voice_builder(context: str):
    # 定义 API 参数
    program = "Chatty_Horo_Voich/VITS-kit/cmd_inference.py"
    api_param_args_1 = "-m" 
    api_param_conf_1 = "/home/xlab-app-center/Elysiavits/G_latest.pth"      #######需要更改
    api_param_args_2 = "-c" 
    api_param_conf_2 = "/home/xlab-app-center/Elysiavits/finetune_speaker.json"       #######需要更改
    api_param_args_3 = "-o" 
    api_param_conf_3 = "/home/xlab-app-center"
    api_param_args_4 = "-l" 
    api_param_conf_4 = "简体中文"
    api_param_args_5 = "-t" 
    api_param_conf_5 = context
    api_param_args_6 = "-s"
    api_param_conf_6 = "Elysia"
    api_param_args_7 = "-ls"
    api_param_conf_7 = "0.85"
    #api_param_args_8 = "-ns"
    #api_param_conf_8 = "0.80"
    #api_param_args_9 = "-nsw"
    #api_param_conf_9 = "0.70"
    
    api_pt = [api_param_args_1, api_param_conf_1, 
              api_param_args_2, api_param_conf_2,
              api_param_args_3, api_param_conf_3,
              api_param_args_4, api_param_conf_4,
              api_param_args_5, api_param_conf_5,
              api_param_args_6, api_param_conf_6,
              api_param_args_7, api_param_conf_7]
              #api_param_args_8, api_param_conf_8,
              #api_param_args_9, api_param_conf_9]
    # 执行另一个 Python 文件，并传递参数
    subprocess.run([sys.executable, program] + api_pt)
    

# 构造模型链的对象
class Chatty_Horo_Chain:
    """
        talk_chain + chatty_chatty + Voicy_Voicy
    """
    def __init__(self, model_path, ):
        self.talk_chain = horowag_conversation_chain(
            llm=Horowag(
                model_path=model_path,
                max_token=16096,
                temperature=0.95,
                top_p=0.40
            )
        )
        
        # Global
        self.ans = None

    def voicy_voicy(self, question: str, chat_history: list = []):
        """
            用于聊天 + 声音
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            self.ans = self.talk_chain.predict(input=question)
            
            # 翻译音频结果
            #translate_ans = self.qwen_translation_chain.run(
                #source_language='中文', 
                #target_language='中文', 
                #text=self.ans
            #)
            
            #print("翻译结果是：", translate_ans)
            # 转化音频文件(时序)
            voice_builder(context=self.ans)
            
            # 聊天函数
            my_list.append(
                (question, self.ans)
            )
         
            chat_history.append(
               " "
            )
            
            return "", chat_history,my_list
        except Exception as e:
            return e, chat_history,my_list

    # Normal Chat
    def chatty_chatty(self, question: str, chat_history: list = []):
        """
            用于聊天
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            self.ans = self.talk_chain.predict(input=question)

            # 聊天函数
            chat_history.append(
                 "..."
            )
            
            return "", chat_history
        except Exception as e:
            return e, chat_history    

    # 音频模块
    def txt_to_audio(self):
        """
            用于音频转化
        """
        # 路径
        return "/home/xlab-app-center/output.wav"


# 构建对话模式
Chatty_Horo_Chain = Chatty_Horo_Chain(
    model_path="Horowag_7b",
    #qwen_translation_chain=qwen_translation_chain
)

# 构建 gradio 对话
block_1 = gr.Blocks()
with block_1 as demo_1:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown(
                """
                <h1>
                <center>Elysia</center>
                </h1>
                <center>ooO 语音 + 文本 Ooo</center>
                """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(height=695, show_copy_button=True)
        with gr.Column(scale=2):
            audiobot = gr.Audio(
                type="filepath",
                interactive=False,
                autoplay=False
            )
            with gr.Row():      
                gr.Image(
                    value="Chatty_Horo_Voich/src/gradio_img/v2-57874c12f9588ffbff916353481e41d3_r.jpg",            ##需修改
                    interactive=False,
                    height="auto",
                    label="Elysia",
                    type="pil"
                )

    with gr.Row(equal_height=True):
        with gr.Column(scale=8):
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="在此输入聊天内容...")
            with gr.Row():
                # 创建提交按钮。
                chatty_btn = gr.Button("Chat")
            with gr.Row():
                 # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                     components=[chatbot], value="Clear console")

        # 设置按钮的点击事件。当点击时，调用上面定义的函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        chatty_btn.click(
            fn=Chatty_Horo_Chain.voicy_voicy,
            inputs=[msg, chatbot],
            outputs=[my_list,msg,chatbot],
        )

        chatbot.change(fn=Chatty_Horo_Chain.txt_to_audio,
                       inputs=None,
                       outputs=[audiobot])

    gr.Markdown("""与Elysia闲聊时的提示：
    <br>
    1. 语音版因为算力限制，运算时间较长(>=20s, <=100s)，请耐心等待
    2. 该版本下，模型对问题的回答会转化为音频，放置于音频输出框内
    <br>
    """)



# threads to consume the request
demo = gr.TabbedInterface(
    [block_1], 
    ["Voicy_Voicy"],
    theme=theme
)



# threads to consume the request
gr.close_all()

# 针对 Gradio的美化

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(share=True, server_port=7860)
