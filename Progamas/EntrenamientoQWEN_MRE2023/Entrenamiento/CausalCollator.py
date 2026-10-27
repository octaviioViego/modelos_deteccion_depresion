import torch

class CausalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len),
                               self.pad_token_id,
                               dtype=torch.long,)
        
        attention_mask = torch.zeros((batch_size, max_len),
                                     dtype=torch.long,)
        
        labels = torch.full((batch_size, max_len),
                            -100,
                            dtype=torch.long,)

        for i, item in enumerate(features):
            n = len(item["input_ids"])

            input_ids[i, :n] = torch.tensor(item["input_ids"], 
                                            dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(item["labels"], 
                                         dtype=torch.long)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,}
