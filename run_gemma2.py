from transformers import AutoTokenizer, AutoModelForCausalLM ,BitsAndBytesConfig
import torch

# 配置 4-bit 量化
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加載量化模型
model = AutoModelForCausalLM.from_pretrained("finetuned_modelTAI", quantization_config=nf4_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("finetuned_modelTAI")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # 調整生成參數
    outputs = model.generate(
        **inputs, 
        max_new_tokens=300,  # 增加生成的最大新詞數
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

while True:
    user_input = input("請輸入您的問題 (輸入 'exit' 以退出): ")
    if user_input.lower() == "exit":
        break
    response = generate_response(user_input)
    print("模型回應: ", response)
