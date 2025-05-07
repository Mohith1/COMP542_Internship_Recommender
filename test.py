import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Load the saved model and tokenizer
model_path = 'model/job_recommendation_model'
tokenizer_path = 'model/job_recommendation_tokenizer'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Create a prediction function
def predict(texts):
    # If only one sentence, make it a list
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy()

# 3. Load your label encoder to decode predictions
import pickle
from sklearn.preprocessing import LabelEncoder

# Re-load your label encoder (assuming you saved it earlier, or else re-fit it)
# Here, we'll re-fit it again for now:
import pandas as pd
df = pd.read_csv('data/UpdatedResumeDataSet.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df['Category'])

# 4. Try predicting!
sample_texts = [
    "aws python machine learning java",
    "social media market research brand strategy ppc adwords hubspot salesforce copywriting"
]

predictions = predict(sample_texts)

# Decode the numeric predictions back to category names
decoded_predictions = label_encoder.inverse_transform(predictions)

for text, pred in zip(sample_texts, decoded_predictions):
    print(f"\nInput Text: {text}\nPredicted Category: {pred}")
