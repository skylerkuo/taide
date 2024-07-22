import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM ,BitsAndBytesConfig

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
def TAIDEchat(sInput):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = (f"生成完成目標 {sInput}的動作\n\n"  #generate actions to complete the goal of {sInput}
              f"必須用如下的格式：\n\n" #must following the format
              f"{{\n"
              f"  \"0\": \"動作\",\n"  #動作 means action
              f"  \"1\": \"動作\",\n"
              f"  \"2\": \"動作\",\n"
              f"  ...\n"
              f"}}\n\n"
              f"不要寫程式 回覆不重複" #do not generate code and answer again
             )
    messages = [
        {"role": "user", "content": prompt}
    ]
    # 將提示詞輸入模型
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # 生成回覆
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # 解碼回覆
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
while 1:
    sInput = input()  
    if sInput == 'exit':
        break
    response = TAIDEchat(sInput)
    print(response)
