# Load training data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load training data
data = pd.read_csv('incidents_train.csv', index_col=0)
train_df, dev_df = train_test_split(data, test_size=0.2, random_state=2024)

# Check text lengths 
print(data.title.str.split().apply(len).describe())

# Import required libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm
import os
from shutil import make_archive

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased' ,  padding='max_length', 
    truncation=True, 
    max_length=128 )

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['title'], padding=True, truncation=True)

# Data preprocessing function
def prepare_data(label): 
    label_encoder = LabelEncoder()
    label_encoder.fit(data[label])

    train_df['label'] = label_encoder.transform(train_df[label])
    dev_df['label'] = label_encoder.transform(dev_df[label])

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=16)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return (
        DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator),
        DataLoader(dev_dataset, batch_size=8, collate_fn=data_collator),
        label_encoder
    )

# Evaluation function
def compute_score(hazards_true, products_true, hazards_pred, products_pred):
    f1_hazards = f1_score(hazards_true, hazards_pred, average='macro')
    f1_products = f1_score(
        products_true[hazards_pred == hazards_true],
        products_pred[hazards_pred == hazards_true],
        average='macro'
    )
    return (f1_hazards + f1_products) / 2.

# Training function
def train_model(model, train_dataloader, dev_dataloader, label_encoder, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Evaluate model
    model.eval()
    total_predictions = []
    with torch.no_grad():
        for batch in dev_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_predictions.extend([p.item() for p in predictions])

    predicted_labels = label_encoder.inverse_transform(total_predictions)
    gold_labels = label_encoder.inverse_transform(dev_df.label.values)
    print(classification_report(gold_labels, predicted_labels, zero_division=0))
    return predicted_labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Subtask 1: Hazard Category
label = 'hazard-category'
train_dataloader, dev_dataloader, le_hazard_category = prepare_data(label)
model_hazard_category = BertForSequenceClassification.from_pretrained(
    'bert-large-uncased', num_labels=len(data[label].unique())
).to(device)

predictions_hazard_category = train_model(
    model_hazard_category, train_dataloader, dev_dataloader, le_hazard_category
)
dev_df['predictions-hazard-category'] = predictions_hazard_category

model_hazard_category.save_pretrained("bert_hazard_category")
np.save("bert_hazard_category/label_encoder.npy", le_hazard_category.classes_)

# Subtask 1: Product Category
label = 'product-category'
train_dataloader, dev_dataloader, le_product_category = prepare_data(label)
model_product_category = BertForSequenceClassification.from_pretrained(
    'bert-large-uncased', num_labels=len(data[label].unique())
).to(device)

predictions_product_category = train_model(
    model_product_category, train_dataloader, dev_dataloader, le_product_category
)
dev_df['predictions-product-category'] = predictions_product_category

model_product_category.save_pretrained("bert_product_category")
np.save("bert_product_category/label_encoder.npy", le_product_category.classes_)

# Subtask 2: Hazard
label = 'hazard'
train_dataloader, dev_dataloader, le_hazard = prepare_data(label)
model_hazard = BertForSequenceClassification.from_pretrained(
    'bert-large-uncased', num_labels=len(data[label].unique())
).to(device)

predictions_hazard = train_model(
    model_hazard, train_dataloader, dev_dataloader, le_hazard
)
dev_df['predictions-hazard'] = predictions_hazard

model_hazard.save_pretrained("bert_hazard")
np.save("bert_hazard/label_encoder.npy", le_hazard.classes_)

# Subtask 2: Product
label = 'product'
train_dataloader, dev_dataloader, le_product = prepare_data(label)
model_product = BertForSequenceClassification.from_pretrained(
    'bert-large-uncased', num_labels=len(data[label].unique())
).to(device)

predictions_product = train_model(
    model_product, train_dataloader, dev_dataloader, le_product
)
dev_df['predictions-product'] = predictions_product

model_product.save_pretrained("bert_product")
np.save("bert_product/label_encoder.npy", le_product.classes_)
tokenizer.save_pretrained("bert_tokenizer")

# Evaluate subtask 2
score = compute_score(
    dev_df['hazard'], dev_df['product'],
    dev_df['predictions-hazard'], dev_df['predictions-product']
)
print(f"Score Subtask 2: {score:.3f}")

# Predict test set
test_df = pd.read_csv('validation_data.csv', index_col=0)

# Prediction function
def predict(texts, model_path):
    tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(model_path + '/label_encoder.npy', allow_pickle=True)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return label_encoder.inverse_transform(predictions.cpu().numpy().tolist())

# Predict test data
predictions = pd.DataFrame()
for column in ['hazard-category', 'product-category', 'hazard', 'product']:
    predictions[column] = predict(test_df.title.tolist(), f"bert_{column.replace('-', '_')}")

# Save predictions
os.makedirs('./submission/', exist_ok=True)
predictions.to_csv('./submission/submission.csv')
make_archive('./submission', 'zip', './submission')
