import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from model import MultiTaskXLMR
from dataset_loader import get_dataloaders

def evaluate_model(
    test_path,
    model_path,
    mode='multi_uncertainty',
    batch_size=8
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Evaluating mode: {mode} on {device}")

    # Load test set
    _, _, test_dl = get_dataloaders(
        train_path=test_path, 
        val_path=test_path, 
        test_path=test_path,
        batch_size=batch_size
    )

    # Load model
    model = MultiTaskXLMR(mode=mode)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_sent_preds, all_sent_labels = [], []
    all_tox_preds, all_tox_labels = [], []
    all_anom_preds, all_anom_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {
                'sentiment_label': batch['sentiment_label'].to(device),
                'toxicity_label': batch['toxicity_label'].to(device),
                'anomaly_label': batch['anomaly_label'].to(device)
            }

            outputs = model(input_ids, attention_mask)

            # Sentiment (3-class)
            sent_probs = F.softmax(outputs['sentiment_logits'], dim=1)
            sent_preds = torch.argmax(sent_probs, dim=1)
            all_sent_preds.extend(sent_preds.cpu().numpy())
            all_sent_labels.extend(labels['sentiment_label'].cpu().numpy())

            # Toxicity (binary)
            tox_probs = torch.sigmoid(outputs['toxicity_logits']).view(-1)
            tox_preds = (tox_probs > 0.5).long()
            all_tox_preds.extend(tox_preds.cpu().numpy())
            all_tox_labels.extend(labels['toxicity_label'].cpu().numpy())

            # Anomaly (binary)
            anom_probs = torch.sigmoid(outputs['anomaly_logits']).view(-1)
            anom_preds = (anom_probs > 0.5).long()
            all_anom_preds.extend(anom_preds.cpu().numpy())
            all_anom_labels.extend(labels['anomaly_label'].cpu().numpy())

    # === Metrics ===
    def binary_metrics(preds, labels, name):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.0
        print(f"📊 {name} → Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        return acc, f1, auc

    def sentiment_metrics(preds, labels):
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"📊 Sentiment → Acc: {acc:.4f} | F1: {f1:.4f}")
        return acc, f1

    print("\n========== EVALUATION RESULTS ==========")
    sent_acc, sent_f1 = sentiment_metrics(all_sent_preds, all_sent_labels)
    tox_acc, tox_f1, tox_auc = binary_metrics(all_tox_preds, all_tox_labels, "Toxicity")
    anom_acc, anom_f1, anom_auc = binary_metrics(all_anom_preds, all_anom_labels, "Anomaly")

    avg_f1 = (sent_f1 + tox_f1 + anom_f1) / 3
    print(f"\n🏁 Average F1 (All Tasks): {avg_f1:.4f}")

    return {
        "sentiment": (sent_acc, sent_f1),
        "toxicity": (tox_acc, tox_f1, tox_auc),
        "anomaly": (anom_acc, anom_f1, anom_auc),
        "average_f1": avg_f1
    }