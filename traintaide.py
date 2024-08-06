import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# https://youtu.be/jh43kY_Gy2I?si=HyY5Xh9FYC-IttkE
# https://www.datacamp.com/tutorial/fine-tuning-large-language-models
# https://huggingface.co/docs/transformers/peft

# quantization, no need if your computer is good enough to run and train the model
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("taide/TAIDE-LX-7B-Chat", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "taide/TAIDE-LX-7B-Chat",
    device_map="auto",
    quantization_config=nf4_config
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

with open(r"C:\Users\User\Desktop\traaindata.txt", 'r') as file:
    lines = file.readlines()

# dataset format : "question = answer"
# example: "3+5=8"

data = []
for line in lines:
    problem, answer = line.strip().split(' = ')
    data.append({'input': problem, 'output': answer})

df= pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

split_dataset = dataset.train_test_split(test_size=0.1)  # 90% train，10% test

def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(targets, truncation=True, padding="max_length", max_length=512)
    model_inputs['labels'] = labels['input_ids'] 
    return model_inputs

# labels是輸出經過轉換的 labels['input_ids']  , model_inputs['labels']是加一個對於輸入的標記 而這個標記就是經過轉換的輸出 這樣可以輸出輸出配成一對

processed_dataset = split_dataset.map(preprocess_function, batched=True) # split dataset 在上面生成的
# map: 對於dataset中的每一個函數使用preprocess_function

training_args = TrainingArguments(
    per_device_train_batch_size=2, # if your computer is good enough, perhaps you can set to 4
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=20,
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained('./finetuned_model2')
tokenizer.save_pretrained('./finetuned_model2')
