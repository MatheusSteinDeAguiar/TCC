import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import pandas as pd
import numpy as np

# Função para calcular as métricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Função para treinar o modelo em um fold
def train_model(train_dataset, eval_dataset, fold_number):
    model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=9)
    training_args = TrainingArguments(
        output_dir=f'./output/results_fold_{fold_number}',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    return model

# Preparar os dados
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

# Função para carregar os dados de um fold
def load_data(fold_idx):
    # Carregar os arquivos de treino e teste para o fold atual
    X_train = pd.read_csv(f'./X_train_fold{fold_idx}.csv')
    y_train = pd.read_csv(f'./y_train_fold{fold_idx}.csv')
    X_test = pd.read_csv(f'./X_test_fold{fold_idx}.csv')
    y_test = pd.read_csv(f'./y_test_fold{fold_idx}.csv')

    # Juntar X e Y em um DataFrame
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Renomear as colunas para 'text' e 'label' (ajuste de acordo com seus dados)
    train_data.columns = ['text', 'label']
    test_data.columns = ['text', 'label']

    return train_data, test_data

# Validação cruzada com k-folds
models = []
for fold_idx in range(1, 11):  # Considerando 10 folds (de 1 a 10)
    # Carregar os dados do fold atual
    train_data, eval_data = load_data(fold_idx)
    
    # Criar os datasets no formato Dataset do Hugging Face
    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)

    # Tokenizar os datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Treinar o modelo no fold atual
    model = train_model(train_dataset, eval_dataset, fold_idx)
    models.append(model)

# Função para fazer predições com múltiplos modelos (voting ensemble)
def predict_with_ensemble(models, dataset):
    all_predictions = []

    # Obter predições de cada modelo
    for model in models:
        trainer = Trainer(model=model)
        predictions = trainer.predict(dataset)
        all_predictions.append(predictions.predictions.argmax(-1))  # Pegar as classes preditas

    # Converter para array para facilitar a manipulação
    all_predictions = np.array(all_predictions)

    # Predição final por votação majoritária (voting)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)
    return final_predictions

# Usar a função de ensemble nos dados de teste
test_preds = predict_with_ensemble(models, test_dataset)

# Salvar as predições
test_data['predictions'] = test_preds
test_data.to_csv('predicoes_no_teste_voting_ensemble.csv', index=False)