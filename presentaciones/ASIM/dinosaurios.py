import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Configuración del dispositivo (GPU si está disponible, vital para Senior devs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextProcessor:
    def __init__(self, text_data):
        # Crear vocabulario de caracteres únicos
        chars = sorted(list(set(text_data)))
        self.vocab_size = (
            len(chars) + 1
        )  # +1 para el token EOS (End of Sequence) si es necesario, o manejamos \n
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}
        self.chars = chars

    def str_to_tensor(self, seq):
        """Convierte string a tensor de índices (LongTensor)"""
        tensor = torch.tensor([self.char_to_ix[ch] for ch in seq], dtype=torch.long).to(
            device
        )
        return tensor


# Datos dummy para probar el código inmediatamente
data_raw = "diplosaurio\ntiranosaurio\nvelociraptor\ntriceratops\nestegosaurio\n"
processor = TextProcessor(data_raw)
