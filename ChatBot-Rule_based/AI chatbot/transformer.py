import json
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

with open('data.json', 'r') as f:
    data = json.load(f)

model_names = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']
models = [AutoModel.from_pretrained(name) for name in model_names]
tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names]

training_data = []
for item in data:
    for pattern in item['patterns']:
        training_data.append((pattern, item['tag']))

class ChatBot(nn.Module):
    def __init__(self):
        super(ChatBot, self).__init__()
        self.hidden_size = sum([model.config.hidden_size for model in models])
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, len(data))
        
    def forward(self, input_ids, attention_mask):
        pooled_outputs = []
        for model, tokenizer in zip(models, tokenizers):
            input_tokens = tokenizer(input_ids, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**input_tokens)
            pooled_outputs.append(outputs.pooler_output)
        pooled_output = torch.cat(pooled_outputs, dim=1)
        x = nn.functional.relu(self.fc1(pooled_output))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

batch_size = 8
learning_rate = 1e-5
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChatBot().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    random.shuffle(training_data)
    for i in range(0, len(training_data), batch_size):
        batch = training_data[i:i+batch_size]
        input_ids = [tokenizer.encode(pattern, add_special_tokens=True) for pattern, tag in batch]
        max_len = max(len(ids) for ids in input_ids)
        input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        attention_mask = (input_ids != 0).to(device)
        targets = torch.tensor([data.index(item[1]) for item in batch], dtype=torch.long).to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def predict_response(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    max_len = max(len(input_ids), 512)
    input_ids = input_ids + [0] * (max_len - len(input_ids))
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = (input_ids != 0).to(device)
    outputs = model(input_ids, attention_mask)
    predicted_index = torch.argmax(outputs).item()
    return data[predicted_index]['responses'][0]

while True:
    text = input('>>')
