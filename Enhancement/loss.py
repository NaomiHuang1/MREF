import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from torch.utils.data import DataLoader

class CustomLoss(nn.Module):
    def __init__(self, model):
        super(CustomLoss, self).__init__()
        self.base_loss_fn = self._get_base_loss_fn(model)

    def _get_base_loss_fn(self, model):
        if hasattr(model, 'num_labels'):
            print("Model num_labels: ", model.num_labels)
            if model.num_labels > 1:
                return nn.CrossEntropyLoss(reduction='none')
            else:
                return nn.BCEWithLogitsLoss(reduction='none')
        else:
            print("Using default CrossEntropyLoss")
            return nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, s_values):
        batch_size = logits.size(0)
        seq_length = logits.size(1)

        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        # Calculate base loss for each token
        base_loss = self.base_loss_fn(logits, labels)

        # Reshape base_loss back to [batch_size, seq_length]
        base_loss = base_loss.view(batch_size, seq_length)

        # Calculate mean loss for each sample (each row in base_loss)
        sample_loss = base_loss.mean(dim=1)  # [batch_size]

        # Ensure s_values is a tensor
        if isinstance(s_values, list):
            s_values = torch.tensor(s_values, device=sample_loss.device, dtype=torch.float)

        # Multiply sample loss by s_values
        weighted_loss = sample_loss * (s_values ** 0.2)

        return weighted_loss.mean()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        model = kwargs.get('model')
        super().__init__(*args, **kwargs)
        self.loss_fn = CustomLoss(model)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        return self.eval_dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        print(f"Debug - Inputs keys: {inputs.keys()}", len(inputs))
        labels = inputs.pop("labels")
        s_values = inputs.pop("s_scores")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.loss_fn(logits, labels, s_values)
        
        return (loss, outputs) if return_outputs else loss
    
    
    
    
