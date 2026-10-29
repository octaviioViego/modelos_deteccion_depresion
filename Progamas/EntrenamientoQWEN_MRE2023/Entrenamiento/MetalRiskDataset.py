
from torch.utils.data import Dataset
from Contexto import enviar_contexto


class MetalRiskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, system_prompt):
        self.examples = []

        for mensaje, etiqueta in zip(
            dataframe["mensaje"],
            dataframe["etiqueta_gold"],
        ):
            ejemplo = enviar_contexto(
                mensaje,
                etiqueta,
                tokenizer,
                system_prompt,
            )
            self.examples.append(ejemplo)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
