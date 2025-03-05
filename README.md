# LUCY: Linguistic Understanding and Control Yielding Early Stage of Her

<p align="center">
    <img src="https://github.com/talkking/LUCY/blob/main/assets/images/3facd97badd54a2380bd1466890ac949_2.png" width="40%" height="20%">
</p>

<p align="center">
    <a href="https://arxiv.org/pdf/2501.16327" target="_blank"><img src="https://img.shields.io/badge/LUCY-Report-b5212f.svg?logo=arxiv" /></a>
    <a href="https://huggingface.co/VITA-MLLM" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" /></a>
    <a href="https://huggingface.co/spaces/shenyunhang/Long-VITA" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-ffc107?color=ffc107&logoColor=white" /></a>
 </p>


## :fire: News
* **`2025.03.05`** 🌟 We have released the code and model checkpoint.
* **`2025.01.27`** 🌟 We are very proud to launch **LUCY**, an end-to-end fully duplex chatbot that supports voice emotion control, tool call, and natural conversation.

## 👀 LUCY Overview
We are excited to present **LUCY**, which incorporates a series of advancements:
1. **Semantic and acoustic emotion control**
2. **Real-time tool call**
3. **Human-like natural conversation**

## 📈 Experimental Results
- **LUCY outperforms professional speech models on ASR benchmarks.**
    <p align="left">
        <img width="634" alt="Clipboard_Screenshot_1741176721" src="https://github.com/user-attachments/assets/7dd6eb8d-4098-42d3-a98b-2acc55b2d039" />
    </p>

- **Emotion Control**
  <p align="left">
      <img width="669" alt="Clipboard_Screenshot_1741176602" src="https://github.com/user-attachments/assets/67968d2b-1de0-4d56-8925-e7dfbba2be09" />
  </p>
- **Tool Call**
  <p align="left">
      <img width="667" alt="Clipboard_Screenshot_1741176644" src="https://github.com/user-attachments/assets/2d60462b-4775-4213-b401-6ae077fd868b" />
  </p>
- **Spoken QA**
  <p align="left">
      <img width="605" alt="Clipboard_Screenshot_1741176675" src="https://github.com/user-attachments/assets/2e58d784-e1d7-4a75-b2e5-037bfac783c6" />
  </p>

## 🔧Usage
### Requirements and Installation
```
git clone https://github.com/VITA-MLLM/LUCY.git
cd LUCY
conda create -n lucy python=3.10 -y
conda activate lucy
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Data Preparation
- An example json file of the training data:
```
[
    ...
    {
        "conversations": [
            {
                "role": "system",
                "content": '''You are an AI robot and your name is Lucy. \n
                - You are a multimodal large language model developed by Tencent. Your aim is to be helpful, honest and harmless. \n
                - You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n               
                - If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.'''
            },
            {
                "role": "user",
                "content": "能帮我写一首现代诗吗？主题是爱情，用点隐喻和象征，再加点自然景物和情感的联系。",
                "wavpath": "mnt/data/hetinggao/Dataset/AudioQA/QA_1450K_question_tar_16khz/question_shuf_part_6/wav/0822474000bf7b3092b44bb226abeffc30f10c09f1_question.wav",
                "codec": "/mnt/data/hetinggao/Dataset/AudioQA/QA_1450K_answer_part1_tar_16khz/answer_part1_shuf_part_3/wav/00514770011b04aa28aa650a5499272436fa3198db_F10.snac"
            },
            {
                "role": "assistant",  
                "content": "松林里，爱情像薄雾，轻轻缠绕。就像清晨的第一缕阳光，在夜晚中无声绽放。秋叶落下，像时光的碎片，我们在微风中分享心事，就像大地和天空的约定。每一滴露珠里，都藏着一个吻。远山如梦，却触手可及，像你在我耳边低语，那声音就像海浪拍打着沙滩，诉说着永恒的爱。爱情像长河，流淌在月光下，每一束月光都闪耀着我们的回忆。无论风雨，我们都是彼此的星光，即使经历风雨，依然坚定。这爱像天地，大地的呼吸，天空的眸光，我们在这自然的旋律中，共舞、相伴。无论风云变幻，我们的心恒久共鸣。"
                "wavpath": "/mnt/data/hetinggao/Dataset/AudioQA/QA_1450K_answer_part1_tar_16khz/answer_part1_shuf_part_3/wav/00514770011b04aa28aa650a5499272436fa3198db_F10.wav",
                "codec": "/mnt/data/hetinggao/Dataset/AudioQA/QA_1450K_question_tar_16khz/question_shuf_part_6/wav/0822474000bf7b3092b44bb226abeffc30f10c09f1_question.snac",
                "speaker": "F10"
            }
        ]
    },
    ...
]
```

### Training
#### Stage 1 
- Aligned audio encoder is available [here](https://huggingface.co/VITA-MLLM/LUCY-Audio-Encoder-110kh).
- Train your own aligned audio encoder, such as **Whisper** Encoder.
    ```
    bash run_scripts/s1.sh
    ```
#### Stage 2 & 3 
- Run the following scripts to continue training stage 2 and stage 3.
    ```
    bash run_scripts/s2p0.sh
    bash run_scripts/s2p1.sh
    bash run_scripts/s3.sh
    ```
### 📐 Inference



# 🤖 Demo
## Demo of Emotion Control
<p align="center">
    <video src="https://github.com/user-attachments/assets/80120730-a37b-4ed5-8da6-7584156a6a67" width="50%" height="50%">
</p>

## Demo of Function Calls
https://github.com/user-attachments/assets/1826bad6-207b-426d-8a99-ce7d684e20f2

## Demo of Natural Conversation
https://github.com/user-attachments/assets/86e9995a-998f-4bbb-8c2e-c51311293cb4

## ✒️ Citation

If you find our work helpful for your research, please consider citing our work. 
```
@article{gao2025lucy,
  title={LUCY: Linguistic Understanding and Control Yielding Early Stage of Her},
  author={Gao, Heting and Shao, Hang and Wang, Xiong and Qiu, Chaofan and Shen, Yunhang and Cai, Siqi and Shi, Yuchen and Xu, Zihan and Long, Zuwei and Zhang, Yike and others},
  journal={arXiv preprint arXiv:2501.16327},
  year={2025}
}
```
