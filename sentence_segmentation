import nltk
from bert_score import score
import warnings
import logging
from transformers import MarianMTModel, MarianTokenizer
import langid
import re

def check_language(string: str) -> str:
    new_string = re.sub(r'[0-9]+', '', string)
    return langid.classify(new_string)[0]

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

model_name = 'Helsinki-NLP/opus-mt-zh-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_zh_to_en(text):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def process_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [word for word in tokens if word.lower() not in ("please", "Please")]
    pos_tags = nltk.pos_tag(tokens)
    
    merged_tokens = []
    current_token = ""
    current_tag = None
    segmented_sentences = []
    current_sentence = []

    for i, (word, tag) in enumerate(pos_tags):
        if current_tag is None:
            current_token = word
            current_tag = tag
        elif tag == current_tag or (current_tag in ('JJ') and tag in ('NN', 'NNS', 'NNP', 'NNPS')):
            current_token += " " + word
            if current_tag in ('JJ') and tag in ('NN', 'NNS', 'NNP', 'NNPS'):
                current_tag = 'NN'
        else:
            merged_tokens.append((current_token, current_tag))
            current_token = word
            current_tag = tag
    
    if current_token:
        merged_tokens.append((current_token, current_tag))

    for i, (token, tag) in enumerate(merged_tokens):
        if tag == 'JJ':
            if (i == 0 or merged_tokens[i-1][1] not in ('NN', 'NNS', 'NNP', 'NNPS')) and \
               (i == len(merged_tokens) - 1 or merged_tokens[i+1][1] not in ('NN', 'NNS', 'NNP', 'NNPS')):
                merged_tokens[i] = (token, 'NN') #如果形容詞不在名詞左右邊 則改為...

    nouns = [token for token, tag in merged_tokens if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
    verbs = [token for token, tag in merged_tokens if tag in ('VB', 'VBD')]

    a=0
    for word, tag in pos_tags:
        if word in verbs:
            a=1
            if current_sentence:
                segmented_sentences.append(" ".join(current_sentence))
            current_sentence = [word]  # Start a new sentence with the verb
        else:
            if a==1:
                current_sentence.append(word)
                
    if current_sentence:
        segmented_sentences.append(" ".join(current_sentence))
    
    return nouns, verbs, segmented_sentences

while True:
    user_command = input("User's command (type 'exit' to quit): ")
    
    if user_command.strip().upper() == 'exit':
        print("Exiting...")
        break

    lang = langid.classify(user_command)
    
    if lang[0] == 'zh':
        translate_eng = translate_zh_to_en(user_command)
        print("Translated Text:", translate_eng[0])
        user_command = translate_eng[0]
        user_command = user_command.replace('.', '')
    
    user_command = user_command.lower()
    
    user_command_nouns, user_command_verbs, segmented_sentences = process_sentence(user_command)
    
    print(f"User command action: {user_command_verbs}")
    print(f"Segmented sentences: {segmented_sentences}")
