import json
import torch
import random
import copy
from model import TranNet

from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = 'intents.json'
with open( file_path, 'r') as json_data:
    intents = json.load(json_data)

FILE = "model/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = TranNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_intent(user_message):
    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.90:
        for intent in intents:
            if tag == intent["tag"]:
                return intent,prob.item()
            
    return {}, prob.item()

def get_bot_response(user_message):
    response = get_intent(user_message)
    return response

print(get_bot_response('hello'))