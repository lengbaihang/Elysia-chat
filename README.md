# Elysia-chat
让爱莉希雅抚慰你的心吧♪

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

# InternStudio 平台中，从本地 clone 一个已有 pytorch 2.0.1 的环境（后续均在该环境执行，若为其他环境可作为参考）
# 进入环境后首先 bash


bash

 conda create --name Elysia  python=3.10 -y
# 激活环境
conda activate Elysia
# 进入家目录 （~的意思是 “当前用户的home路径”）
cd ~

# 创建一个微调 oasst1 的工作路径，进入
mkdir ~/ft-oasst1 && cd ~/ft-oasst1



# 拉取 0.1.9 的版本源码
git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'



数据准备
创建data文件夹用于存放用于训练的数据集

mkdir -p /root/ft-oasst1/data && cd /root/ft-oasst1/data


参考官方文档https://github.com/InternLM/xtuner/blob/main/docs/en/user_guides/dataset_format.md

[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]

在data目录下新建一个generate_data.py文件，将以下代码复制进去，然后运行该脚本即可生成数据集。

参考：

import json

def convert_lines_to_dataset(lines):
    dataset = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            dataset.append({
                "conversation": [
                    {
                        "system": "",
                        "input": "",
                        "output": line
                    }
                ]
            })
    return dataset

# 读取文件并转换为数据集
def file_to_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    dataset = convert_lines_to_dataset(lines)
    return dataset

# 使用函数
filename = '/root/personal_assistant/data/a.txt'  # 替换为你的文件名
dataset = file_to_dataset(filename)

# 将数据集转换为JSON字符串
json_str = json.dumps(dataset, ensure_ascii=False, indent=4)

print(json_str)


with open('personal_assistant.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)





这里进行大量的增量微调，构建爱莉的语气与知识背景





配置准备

下载模型InternLM-chat-7B

# 创建一个目录，放模型文件，防止散落一地
mkdir ~/ft-oasst1/internlm-chat-7b

# 装一下拉取模型文件要用的库
pip install modelscope

# 从 modelscope 下载下载模型文件
cd ~/ft-oasst1
apt install git git-lfs -y
git lfs install
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm-chat-7b.git -b v1.0.3

由于 huggingface 网络问题，将以下指令复制到正确位置即可(教学平台)：

cd ~/ft-oasst1
# ...-guanaco 后面有个空格和英文句号啊
cp -r /root/share/temp/datasets/openassistant-guanaco .

TIP:如果有问题可以先同步一下仓库（我当时遇到了问题，但没人提出）和升级pip



对于其他微调方法，建议采用以下命令获取列表，查找相关文件：

xtuner list-cfg -p internlm
其中-p 为模糊查找，若想训练其他模型，可以修改 internlm 为 Xtuner 支持的其他模型名称。如果所提供的配置文件不能满足使用需求，请导出所提供的配置文件并进行相应更改：

xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
例如通过下列命令将名为 internlm_7b_qlora_oasst1_e3 的 config 导出至当前目录下：



xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .


修改拷贝后的文件internlm_chat_7b_qlora_oasst1_e3_copy.py，修改下述位置：

# PART 1 中
# 预训练模型存放的位置
pretrained_model_name_or_path = '/root/ft-oasst1/internlm-chat-7b'

# 微调数据存放的位置
data_path = '/root/ft-oasst1/data/personal_assistant.json'

# 训练中最大的文本长度
max_length = 512

# 每一批训练样本的大小
batch_size = 2

# 最大训练轮数
max_epochs = 3

# 验证的频率
evaluation_freq = 90

# 用于评估输出内容的问题（用于评估的问题尽量与数据集的question保持一致）
evaluation_inputs = [ '请介绍一下你自己', '请做一下自我介绍' ]    #按实际情况修改


# PART 3 中
dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))
dataset_map_fn=None


微调启动

用xtuner train命令启动训练、

xtuner train /root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3.py  --deepspeed deepspeed_zero2

微调后参数转换/合并

训练后的pth格式参数转Hugging Face格式

# 创建用于存放Hugging Face格式参数的hf文件夹
mkdir /root/ft-oasst1/work_dirs/hf

export MKL_SERVICE_FORCE_INTEL=1

# 配置文件存放的位置
export CONFIG_NAME_OR_PATH=/root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3_copy.py

# 模型训练后得到的pth格式参数存放的位置
export PTH=/root/ft-oasst1/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_3.pth

# pth文件转换为Hugging Face格式后参数存放的位置
export SAVE_PATH=/root/ft-oasst1/work_dirs/hf

# 执行参数转换
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

# 执行参数Merge
xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
    --max-shard-size 2GB

网页DEMO

pip install streamlit==1.24.0
# 创建code文件夹用于存放InternLM项目代码
mkdir /root/ft-oasst1/code && cd /root/ft-oasst1/code
git clone https://github.com/InternLM/InternLM.git

切换 commit 版本，与教程 commit 版本保持一致，可以让大家更好的复现。

cd InternLM
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17


将 /root/code/InternLM/web_demo.py 中 29 行和 33 行的模型路径更换为Merge后存放参数的路径 /root/personal_assistant/config/work_dirs/hf_merge

运行 /root/personal_assistant/code/InternLM 目录下的 web_demo.py 文件，将端口映射到本地。在本地浏览器输入 http://127.0.0.1:6006 即可。

streamlit run /root/personal_assistant/code/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006

效果还是不错

![IMG_0493(20240224-191912)](https://github.com/lengbaihang/Elysia-chat/assets/96370602/c023bfb0-1a25-4b4b-9343-40e2b2a04f2b)

![IMG_0494(20240224-191915)](https://github.com/lengbaihang/Elysia-chat/assets/96370602/eab7f06d-94a5-46e2-87fe-3e766032ee8a)





tips；0.1版本只用了100句话，调用的时候，需要先用台词喂一句，才能有不错的结果


未来展望
  0.2 重大更新中 优化数据集，计划增加LangChain 搭建爱莉的知识库

  
  0.3 计划增加LangChain 搭建爱莉的知识库增加语音（小助手需要机遇算力支持）



  
  0.4 计划增加 Agent框架,使爱莉无所不能哦

  0.5 加入  梅比乌斯
            伊甸
            樱
            帕朵
            格蕾修
            华
            月下


  0.6 修建多人物共同对话——————————黄金庭院茶话会
            


  0.7 计划增加新人物     1. Vivy -Fluorite Eye’s Song-    vivy    https://baike.baidu.com/item/%E8%96%87%E8%96%87%20-%E8%90%A4%E7%9F%B3%E7%9C%BC%E4%B9%8B%E6%AD%8C-/56618413
                        2. BEATLESS                      蕾西亚   https://baike.baidu.com/item/%E8%95%BE%E8%A5%BF%E4%BA%9A/22329569?fromModule=lemma_inlink      
                        Vivy 我是你的粉丝呀
                        蕾西亚赛高，茶道赛高，咕噜咕噜
  
    





