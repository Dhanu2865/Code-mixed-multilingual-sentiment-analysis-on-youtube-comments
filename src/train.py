import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from model import MultiTaskXLMR
from dataset_loader import get_dataloaders

def train_model(
    train_path,
    val_path,
    test_path,
    mode='multi_uncertainty',
    save_dir='results/weights',
    num_epochs=5,
    batch_size=8,
    lr=2e-5,
    warmup_ratio=0.1
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training mode: {mode} on device: {device}")

    # === Load data ===
    train_dl, val_dl, _ = get_dataloaders(train_path, val_path, test_path, batch_size=batch_size)

    # === Model, optimizer, scheduler ===
    model = MultiTaskXLMR(mode=mode)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')

    # === Training loop ===
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {
                'sentiment_label': batch['sentiment_label'].to(device),
                'toxicity_label': batch['toxicity_label'].to(device),
                'anomaly_label': batch['anomaly_label'].to(device)
            }

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dl)
        print(f"🟢 Avg Train Loss: {avg_train_loss:.4f}")

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = {
                    'sentiment_label': batch['sentiment_label'].to(device),
                    'toxicity_label': batch['toxicity_label'].to(device),
                    'anomaly_label': batch['anomaly_label'].to(device)
                }
                outputs = model(input_ids, attention_mask, labels)
                total_val_loss += outputs['loss'].item()

        avg_val_loss = total_val_loss / len(val_dl)
        print(f"🔵 Avg Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # ✅ fixed absolute save path in Drive
            save_dir = "/content/drive/MyDrive/multitask_xlmr/results/weights"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"best_model_{mode}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")


    print("✅ Training complete!")