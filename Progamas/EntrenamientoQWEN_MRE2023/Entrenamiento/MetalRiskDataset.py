from Progamas.EntrenamientoQWEN_MRE2023.Entrenamiento.Contexto import enviar_contexto
from torch.utils.data import Dataset

class MetalRiskDataset(Dataset):
    
    def __init__(self, dataframe):
        self.examples = []
        LABEL_COLUMN = "etiqueta_gold"
        TEXT_COLUMN = "mensaje"
        
        for mensaje, etiqueta in zip(dataframe[TEXT_COLUMN],dataframe[LABEL_COLUMN],):
            self.examples.append(enviar_contexto(mensaje, etiqueta))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
