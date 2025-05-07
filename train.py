import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(num_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return tokenizer, model

# Load the new dataset
df = pd.read_csv('/Resume1.csv')

# Clean and extract text and labels
def clean_text(text):
    return text.replace('\r', ' ').replace('\n', ' ').replace('â¢', ' ').strip()

texts = [clean_text(t) for t in df['Resume'].tolist()]
labels = df['Category'].tolist()

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 4. Find number of classes
num_classes = len(set(labels))

# 5. Load tokenizer and model with correct number of labels
tokenizer, model = load_model(num_labels=num_classes)

# 6. Create dataset and dataloader
dataset = ResumeDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 7. Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 8. Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 9. Training loop
model.train()
for epoch in range(3):  # You can change the number of epochs
    running_loss = 0.0
    loop = tqdm(dataloader, leave=True)  # Create a tqdm progress bar
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update tqdm bar description
        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}')

# 10. Save model and tokenizer
model.save_pretrained('model/job_recommendation_model')
tokenizer.save_pretrained('model/job_recommendation_tokenizer')
print("Model saved successfully!")