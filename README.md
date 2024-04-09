# Elysia-chat
让爱莉希雅抚慰你的心吧♪



[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/apps/detail/lengbaihang1/Elysia) ![Static Badge](https://img.shields.io/badge/license-Apache--2.0-green?label=license)

[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/usercenter/lengbaihang1?vtab=create&module=models)

![v2-53f9a387bba7b2f673ffb242934afdc3_r](https://github.com/lengbaihang/Elysia-chat/assets/96370602/27967b2e-8ccb-410a-9c21-ee150f5e65b3)
![d2f9d3e7d0eb80f423a441430db69446_5839222736512585157](https://github.com/lengbaihang/Elysia-chat/assets/96370602/c0221a2d-c07d-4ebe-89d3-a76f1d516a85)
![v2-57874c12f9588ffbff916353481e41d3_r](https://github.com/lengbaihang/Elysia-chat/assets/96370602/0ccbd8b2-af8e-4bc3-954b-cbbe9a50b58d)

这是我的一小步，却是爱门的一大步

              *Introduction**


| 基础属性 | 模型相关解释 |
|:-------:|:-------:|
| 模型基座 | InternLM2-Chat-7b |
| 微调方法 | QLoRA |
| 技术库 | Xtuner + Transformers |


参考并学习大佬的经验： https://github.com/InternLM/tutorial/blob/main/xtuner/self.md

                     https://github.com/InternLM/tutorial/blob/main/xtuner/README.md
                     
                     https://github.com/SaaRaaS-1300/InternLM_openNotebook/tree/main/Horowag_7b
                     
                    https://datawhalechina.github.io/llm-universe/#/
                    
                    https://github.com/InternLM/xtuner
                    
                    https://github.com/Plachtaa/VITS-fast-fine-tuning/blob/main/README_ZH.md
                    



tips:书生大模型的在线实验室创建虚拟环境好像没啥用，无法隔离文件与依赖，所以要注意一点（能白嫖A100算力还想什么自行车！想白嫖的小伙伴快来）
      

微调环境准备（这里非常标准的步骤）

# InternStudio 平台




bash

     conda create --name Elysia  python=3.10 -y


    conda activate Elysia


      cd ~



    mkdir ~/ft-oasst1 && cd ~/ft-oasst1




    git clone -b v0.1.9  https://github.com/InternLM/xtuner
    
     git clone -b v0.1.9 https://gitee.com/Internlm/xtuner


    cd xtuner

    python -m pip install --upgrade pip


    pip install -e '.[all]'



数据准备


    mkdir -p /root/ft-oasst1/data && cd /root/ft-oasst1/data


参考官方文档https://github.com/InternLM/xtuner/blob/main/docs/en/user_guides/dataset_format.md



在data目录下新建一个generate_data.py文件，将以下代码复制进去，然后运行该脚本即可生成数据集。







这里进行大量的增量微调，构建爱莉的语气与知识背景





    mkdir ~/ft-oasst1/internlm-chat-7b





    cd ~/ft-oasst1
    apt install git git-lfs -y
    git lfs install
    git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm-chat-7b.git -b v1.0.3

由于 huggingface 网络问题，将以下指令复制到正确位置即可(教学平台)：

    cd ~/ft-oasst1

    cp -r /root/share/temp/datasets/openassistant-guanaco .

TIP:如果有问题可以先同步一下仓库（我当时遇到了问题，但没人提出）和升级pip



对于其他微调方法，建议采用以下命令获取列表，查找相关文件：

    xtuner list-cfg -p internlm
其中-p 为模糊查找，若想训练其他模型，可以修改 internlm 为 Xtuner 支持的其他模型名称。如果所提供的配置文件不能满足使用需求，请导出所提供的配置文件并进行相应更改：

    xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
例如通过下列命令将名为 internlm_7b_qlora_oasst1_e3 的 config 导出至当前目录下：



    xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .


修改拷贝后的文件internlm_chat_7b_qlora_oasst1_e3_copy.py，修改下述位置：


    pretrained_model_name_or_path = '/root/ft-oasst1/internlm-chat-7b'


    data_path = '/root/ft-oasst1/data/personal_assistant.json'


    max_length = 1024


    batch_size = 2


    max_epochs = 3


    evaluation_freq = 5


    evaluation_inputs = [ '请介绍一下你自己', '请做一下自我介绍' ]    #按实际情况修改



    dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))
    dataset_map_fn=None




微调启动

用xtuner train命令启动训练、

    xtuner train /root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3.py  --deepspeed deepspeed_zero2






使用tmux

    apt update -y

    apt install tmux -y

    tmux new -s Elysia

    tmux attach -t Elysia
    退出ctrl B +D


    
    

微调后参数转换/合并

训练后的pth格式参数转Hugging Face格式


    mkdir /root/ft-oasst1/work_dirs/hf

    

    export MKL_SERVICE_FORCE_INTEL=1
    


    export CONFIG_NAME_OR_PATH=/root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3.py


    export PTH=/root/ft-oasst1/work_dirs/internlm_chat_7b_qlora_oasst1_e3/epoch_3.pth
    
  
      export SAVE_PATH=/root/ft-oasst1/work_dirs/hf

 执行参数转换
 
    xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH

Merge模型参数

   

     export MKL_SERVICE_FORCE_INTEL=1
     export MKL_THREADING_LAYER='GNU'
     # 原始模型参数存放的位置
     export NAME_OR_PATH_TO_LLM=/root/ft-oasst1/internlm-chat-7b
     # Hugging Face格式参数存放的位置
     export NAME_OR_PATH_TO_ADAPTER=/root/ft-oasst1/work_dirs/hf
     # 最终Merge后的参数存放的位置
     mkdir /root/ft-oasst1/work_dirs/hf_merge
     export SAVE_PATH=/root/ft-oasst1/work_dirs/hf_merge
     xtuner convert merge \
        $NAME_OR_PATH_TO_LLM \
        $NAME_OR_PATH_TO_ADAPTER \
        $SAVE_PATH \
        --max-shard-size 2GB



    

 网页DEMO

    pip install streamlit==1.24.0
 创建code文件夹用于存放InternLM项目代码
 
    mkdir /root/ft-oasst1/code && cd /root/ft-oasst1/code
    git clone https://github.com/InternLM/InternLM.git

切换 commit 版本，与教程 commit 版本保持一致，可以让大家更好的复现。

    cd InternLM
    git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17


 将 /root/ft-oasst1/code/InternLM/web_demo.py 中 29 行和 33 行的模型路径更换为Merge后存放参数的路径 /root/ft-oasst1/work_dirs/hf_merge

 运行 /root/personal_assistant/code/InternLM 目录下的 web_demo.py 文件，将端口映射到本地。在本地浏览器输入 http://127.0.0.1:6006 即可。

    ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 37560


    streamlit run /root/ft-oasst1/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006

    
效果还是不错








  0.2 （now）已优化数据集，增加语音，增加LangChain，支持多轮对话。

  
  0.3 计划增加LangChain 搭建爱莉的知识库（小助手需要机遇算力支持）



  
  0.4 计划增加 Agent框架,使爱莉无所不能哦

  

  

  0.5 加入  梅比乌斯
            伊甸
            樱
            帕朵
            格蕾修
            华
            月下

            


  0.6 修建多人物共同对话——————————黄金庭院茶话会

  
            


  0.7 计划增加新人物    
  
1. Vivy -Fluorite Eye’s Song-    vivy    https://baike.baidu.com/item/%E8%96%87%E8%96%87%20-%E8%90%A4%E7%9F%B3%E7%9C%BC%E4%B9%8B%E6%AD%8C-/56618413


  
2. BEATLESS                      蕾西亚   https://baike.baidu.com/item/%E8%95%BE%E8%A5%BF%E4%BA%9A/22329569?fromModule=lemma_inlink




                        
                        Vivy 我是你的粉丝呀
                        蕾西亚赛高，茶道赛高，咕噜咕噜
  
    





