import torch


class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: CrossEntropyLoss between corresponding logits and tokens
        """
        pass
    

class LMAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, tokens, tokens_lens, loss_mask=None):
        """
        :param torch.Tensor outputs: Output from LM.forward. Shape: [B, T, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [B, T]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :param torch.Tensor loss_mask: Valid next-token transitions. Shape: [B, T - 1]
        :return torch.Tensor: Accuracy for given logits and tokens
        """
        pass
