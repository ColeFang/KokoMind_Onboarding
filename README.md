# KokoMind_Onboarding

This repo contains three parts for the onboarding part of [Kokomind project](https://chats-lab.github.io/KokoMind/).

## Content

1. [Requirements](#requirements)
2. [Run through RLHF](#Run-through-RLHF)    
3. [Finetuning llama2-based paraphrase model with peft ](#Finetuning-llama2)
4. [automatic_prompt_engineer](#automatic_prompt_engineer)
5. [Ghost Attention in llama2](#Ghost-Attention-in-llama2)
6. [Contact](#contact)

## Requirements

```
conda create -n llama python=3.10.12
conda activate llama
pip3 install -r requirements.txt --user
```

## Run through RLHF
### [Source code](https://github.com/CarperAI/trlx) 

The process of RLHF can be separated in three parts:
<p align="center">
  <img src="img/animation.gif" width="70%" height="70%">
</p>

1. install trxl with command:
```
cd scripts/trxl
pip install -e .
```

2. Fine-tune with Supervision (SFT) running with:
```
!deepspeed examples/summarize_rlhf/sft/train_gptj_summarize.py
```

3. Train the Reward Model running with:
```
!deepspeed examples/summarize_rlhf/reward_model/train_reward_model_gptj.py
```

4. Fine-tune with PPO running with:
```
!deepspeed examples/summarize_rlhf/trlx_gptj_text_summarization.py
```
## Finetuning llama2-based paraphrase model with peft 

### [Source code](https://github.com/huggingface/trl) 
using command:
```
cd scripts/peft
python llama_peft.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_4bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```


## Contact

Please leave Github issues or contact Hongchao Fang at `fang.hong@northeastern.edu` for any questions.
