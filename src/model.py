import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', mode='multi_equal', hidden_size=768, num_sentiment=3):
        """
        mode: 'single', 'multi_equal', or 'multi_uncertainty'
        """
        super().__init__()
        self.mode = mode
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        # === Task Heads ===
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_sentiment)
        )
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # === For Uncertainty Weighting ===
        if mode == 'multi_uncertainty':
            self.log_var_sent = nn.Parameter(torch.zeros(()))
            self.log_var_tox = nn.Parameter(torch.zeros(()))
            self.log_var_anom = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Returns logits and optionally total loss (if labels provided).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token output

        # === Compute logits ===
        sent_logits = self.sentiment_head(pooled)
        tox_logits = self.toxicity_head(pooled)
        anom_logits = self.anomaly_head(pooled)

        result = {
            "sentiment_logits": sent_logits,
            "toxicity_logits": tox_logits,
            "anomaly_logits": anom_logits
        }

        # === Compute Loss if labels are provided ===
        if labels is not None:
            loss_fn_sent = nn.CrossEntropyLoss()
            loss_fn_bin = nn.BCEWithLogitsLoss()

            loss_sent = loss_fn_sent(sent_logits, labels['sentiment_label'])
            loss_tox = loss_fn_bin(tox_logits.view(-1), labels['toxicity_label'])
            loss_anom = loss_fn_bin(anom_logits.view(-1), labels['anomaly_label'])

            if self.mode == 'single':
                result['loss'] = loss_sent + loss_tox + loss_anom  # simplified if training one head at a time

            elif self.mode == 'multi_equal':
                # equal weighted sum
                result['loss'] = (loss_sent + loss_tox + loss_anom) / 3

            elif self.mode == 'multi_uncertainty':
                # homoscedastic uncertainty weighting (Kendall et al., 2018)
                def weighted(loss, log_var):
                    precision = torch.exp(-log_var)
                    return precision * loss + log_var

                loss_total = 0.5 * weighted(loss_sent, self.log_var_sent) + \
                             0.5 * weighted(loss_tox, self.log_var_tox) + \
                             0.5 * weighted(loss_anom, self.log_var_anom)

                result['loss'] = loss_total

        return result