import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, get_scheduler
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from torch.optim import AdamW

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    quantization_config=nf4_config
)

target_modules = ["q_proj", "v_proj"]
# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules
)


model = get_peft_model(model, lora_config)

# 讀取生成的訓練數據
with open(r"C:\Users\User\Desktop\try11.txt", encoding='utf-8') as file:
    lines = file.readlines()

# 初始化變量
data = []
question, answer = '', ''
collecting_answer = False

# 解析檔案內容
for line in lines:
    line = line.strip()

    if line.startswith("Question:"):
        if question and answer:  # 如果已有上一個問答對，則先保存
            data.append({'input': question, 'output': answer})
        question = line.split(":", 1)[1].strip()  # 提取問題
        answer = ''  # 重置答案
        collecting_answer = True  # 開始收集答案
    elif line.startswith("Answer:"):
        collecting_answer = True  # 開始收集答案
    else:
        if collecting_answer:
            answer += ' ' + line  # 多行答案拼接

# 最後一個問答對加入到數據集中
if question and answer:
    data.append({'input': question, 'output': answer})

# 將數據轉換為DataFrame格式
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# 分割數據集為訓練集和測試集
split_dataset = dataset.train_test_split(test_size=0.1)

# 對訓練集進行洗牌
split_dataset['train'] = split_dataset['train'].shuffle(seed=42)
split_dataset['test'] = split_dataset['test'].shuffle(seed=42)


# 預處理函數
def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 預處理數據集
processed_dataset = split_dataset.map(preprocess_function, batched=True)

# 訓練參數
training_args = TrainingArguments(
    per_device_train_batch_size=1, # 如果電腦性能足夠，batch size 可以設置更大
    gradient_accumulation_steps=8, # 累積8步的梯度
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=20,
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 計算總步數以用於學習率調度器
num_training_steps = training_args.num_train_epochs * len(processed_dataset['train'])
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 初始化 Trainer 並開始訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
    tokenizer=tokenizer,
    optimizers=(optimizer, lr_scheduler)
)

trainer.train()

# 保存微調過的模型
model.save_pretrained('./finetuned_modelTAI')
tokenizer.save_pretrained('./finetuned_modelTAI')
