import torch

if torch.cuda.is_available():
    print("Numero di GPU disponibili:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA non disponibile. Nessuna GPU trovata.")