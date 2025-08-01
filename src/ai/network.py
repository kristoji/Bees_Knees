import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Any # Aggiunto per i type hint
from ai.training import Training
from ai.loader import NpzDataset,ConvDataset 
from tqdm import tqdm
import time

BOARD_HEIGHT: int = Training.SIZE 
BOARD_WIDTH: int = Training.SIZE 

NUM_UNIQUE_PIECES: int = Training.NUM_PIECES 
NUM_STACK_LEVELS: int = Training.LAYERS

# Canali totali per l'input della CNN 2D:
INPUT_CHANNELS: int = NUM_UNIQUE_PIECES * NUM_STACK_LEVELS 

# Output della Policy: un vettore di probabilità con la stessa dimensionalità "28x7x56x56"
POLICY_OUTPUT_SIZE: int = NUM_UNIQUE_PIECES * NUM_STACK_LEVELS * BOARD_HEIGHT * BOARD_WIDTH # 28 * 7 * 56 * 56 = 615424

# Parametri dell'architettura della rete (stile AlphaGo Zero)
NUM_RESIDUAL_BLOCKS: int = 19 # Per una torre da 20 blocchi (1 blocco conv iniziale + 19 blocchi residui)
NUM_FILTERS: int = 256        # Numero di filtri nei blocchi convoluzionali e residui

class ConvolutionalBlock(nn.Module):
    """
    Un blocco costituito da una Convoluzione, Batch Normalization e attivazione ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super(ConvolutionalBlock, self).__init__()
        padding: int = (kernel_size - 1) // 2 # Padding per mantenere le dimensioni HxW
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ResidualBlock(nn.Module):
    """
    Un blocco residuale come definito in AlphaGo Zero.
    """
    def __init__(self, num_channels: int, kernel_size: int = 3) -> None:
        super(ResidualBlock, self).__init__()
        padding: int = (kernel_size - 1) // 2
        self.conv1: nn.Conv2d = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(num_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(num_channels)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Connessione Skip
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module):
    """
    Rete neurale dual residual per Hive, basata su AlphaGo Zero, in PyTorch.
    Input: (N, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
           dove INPUT_CHANNELS = NUM_UNIQUE_PIECES * NUM_STACK_LEVELS
    Output: (policy_logits, value_output)
            policy_logits: (N, POLICY_OUTPUT_SIZE)
            value_output: (N, 1)
    """
    def __init__(self, input_channels: int = INPUT_CHANNELS, num_residual_blocks: int = NUM_RESIDUAL_BLOCKS,
                 num_filters: int = NUM_FILTERS, policy_output_size: int = POLICY_OUTPUT_SIZE,
                 board_height: int = BOARD_HEIGHT, board_width: int = BOARD_WIDTH) -> None:
        super(NeuralNetwork, self).__init__()
        
        # --- Residual Tower ---
        self.initial_conv_block: ConvolutionalBlock = ConvolutionalBlock(input_channels, num_filters, kernel_size=3)
        self.residual_tower: nn.Sequential = nn.Sequential(
            *[ResidualBlock(num_filters, kernel_size=3) for _ in range(num_residual_blocks)]
        )
        
        # --- Policy Head ---
        self.policy_conv: nn.Conv2d = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.policy_bn: nn.BatchNorm2d = nn.BatchNorm2d(2)
        policy_fc_input_features: int = 2 * board_height * board_width
        self.policy_fc: nn.Linear = nn.Linear(policy_fc_input_features, policy_output_size)
        nn.init.kaiming_normal_(self.policy_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.policy_fc.weight, mode='fan_out', nonlinearity='relu')
        if self.policy_fc.bias is not None:
            nn.init.zeros_(self.policy_fc.bias)

        # --- Value Head ---
        self.value_conv: nn.Conv2d = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_bn: nn.BatchNorm2d = nn.BatchNorm2d(1)
        value_fc1_input_features: int = 1 * board_height * board_width
        self.value_fc1: nn.Linear = nn.Linear(value_fc1_input_features, 256) # Layer nascosto
        self.value_fc2: nn.Linear = nn.Linear(256, 1) # Output scalare
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_fc1.weight, mode='fan_out', nonlinearity='relu')
        if self.value_fc1.bias is not None:
            nn.init.zeros_(self.value_fc1.bias)
        nn.init.xavier_normal_(self.value_fc2.weight) # Xavier/Glorot per output tanh
        if self.value_fc2.bias is not None:
            nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Residual Tower
        x = self.initial_conv_block(x)
        tower_output: torch.Tensor = self.residual_tower(x)

        # Policy Head
        policy_x: torch.Tensor = self.policy_conv(tower_output)
        policy_x = self.policy_bn(policy_x)
        policy_x = F.relu(policy_x)
        policy_x = torch.flatten(policy_x, start_dim=1) 
        policy_logits: torch.Tensor = self.policy_fc(policy_x)

        # Value Head
        value_x: torch.Tensor = self.value_conv(tower_output)
        value_x = self.value_bn(value_x)
        value_x = F.relu(value_x)
        value_x = torch.flatten(value_x, start_dim=1)
        value_x = F.relu(self.value_fc1(value_x))
        value_output: torch.Tensor = torch.tanh(self.value_fc2(value_x)) 
        
        return policy_logits, value_output
    

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        # print(f"Modello salvato in {path}")

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))
        # print(f"Modello caricato da {path}")

    # def train_epoch(self,
    #                 dataloader: DataLoader, 
    #                 optimizer: optim.Optimizer, 
    #                 policy_criterion: nn.Module, 
    #                 value_criterion: nn.Module, 
    #                 device: torch.device, 
    #                 value_loss_weight: float = 1.0) -> Tuple[float, float, float]:
    #     # model.train()
    #     self.train()
    #     total_loss: float = 0.0
    #     total_policy_loss: float = 0.0
    #     total_value_loss: float = 0.0

    #     # for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
    #     for batch_idx, (states, policy_targets, value_targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training Batches"):
    #         states: torch.Tensor = states.to(device)
    #         policy_targets: torch.Tensor = policy_targets.to(device)
    #         value_targets: torch.Tensor = value_targets.to(device)

    #         optimizer.zero_grad()
    #         policy_logits, value_preds = self(states) #model

    #         # OLD:
    #         # loss_policy: torch.Tensor = policy_criterion(policy_logits, policy_targets)
    #         # ATTEMPT ALPHA ZERO
    #         # log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
    #         # loss_policy = policy_criterion(log_probs, policy_targets)  # policy_targets = distribuzione

    #         log_probs = F.log_softmax(policy_logits, dim=1)                  # [B, P]
    #         policy_targets_flat = policy_targets.view(log_probs.size(0), -1) # [B, P]
    #         # loss_policy = policy_criterion(log_probs, policy_targets_flat)

    #         eps = 1e-8
    #         targets = policy_targets_flat + eps
    #         targets = targets / targets.sum(dim=1, keepdim=True)
    #         policy_criterion = nn.KLDivLoss(reduction="batchmean")
    #         loss_policy = policy_criterion(log_probs, targets)


    #         loss_value: torch.Tensor = value_criterion(value_preds, value_targets.unsqueeze(1).float())
    #         combined_loss: torch.Tensor = loss_policy + value_loss_weight * loss_value

    #         combined_loss.backward()
    #         optimizer.step()

    #         total_loss += combined_loss.item()
    #         total_policy_loss += loss_policy.item()
    #         total_value_loss += loss_value.item()

    #         if batch_idx > 0 and batch_idx % 100 == 0:
    #             print(f"  Batch {batch_idx}/{len(dataloader)}: "
    #                 f"Loss Totale: {combined_loss.item():.4f} (Policy: {loss_policy.item():.4f}, Value: {loss_value.item():.4f})")

    #     avg_loss: float = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    #     avg_policy_loss: float = total_policy_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    #     avg_value_loss: float = total_value_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    #     return avg_loss, avg_policy_loss, avg_value_loss
    
    def train_epoch(self,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                policy_criterion: nn.Module,
                value_criterion: nn.Module,
                device: torch.device,
                value_loss_weight: float = 1.0) -> Tuple[float, float, float]:
        self.train()
        total_loss = total_policy_loss = total_value_loss = 0.0

        for batch_idx, (states, policy_targets, value_targets) in tqdm(enumerate(dataloader),
                                                                    total=len(dataloader),
                                                                    desc="Training Batches"):
            start_batch = time.time()
            # Move data to device
            t0 = time.time()
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            t1 = time.time()
            # print(f"Batch {batch_idx}: Data transfer time: {t1 - t0:.4f}s | states.shape={states.shape}")

            optimizer.zero_grad()

            # Forward pass
            t2 = time.time()
            policy_logits, value_preds = self(states)
            t3 = time.time()
            # print(f"Batch {batch_idx}: Forward time: {t3 - t2:.4f}s | policy_logits.shape={policy_logits.shape}, value_preds.shape={value_preds.shape}")

            # Compute policy loss (KL divergence)
            t4 = time.time()
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_targets_flat = policy_targets.view(log_probs.size(0), -1)

            # # 🔍 DEBUG SNIPPET
            # if torch.isnan(policy_targets_flat).any():
            #     print(f"⚠️ Batch {batch_idx}: policy_targets_flat contiene NaN!")

            # if (policy_targets_flat.sum(dim=1) == 0).any():
            #     print(f"⚠️ Batch {batch_idx}: target con somma zero!")

            # if torch.isnan(log_probs).any():
            #     print(f"⚠️ Batch {batch_idx}: log_probs contiene NaN!")

            # print(f"Batch {batch_idx}: policy_targets_flat[0][:10] = {policy_targets_flat[0][:10]}")
            # print(f"Batch {batch_idx}: policy_targets_flat[0] sum = {policy_targets_flat[0].sum().item():.6f}")

            eps = 1e-8
            targets = policy_targets_flat 
            targets = targets / targets.sum(dim=1, keepdim=True)
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            loss_policy = kl_loss(log_probs, targets)
            t5 = time.time()
            # print(f"Batch {batch_idx}: Policy loss time: {t5 - t4:.4f}s | loss_policy={loss_policy.item():.4f}")

            # Compute value loss
            t6 = time.time()
            loss_value = value_criterion(value_preds, value_targets.unsqueeze(1).float())
            t7 = time.time()
            # print(f"Batch {batch_idx}: Value loss time: {t7 - t6:.4f}s | loss_value={loss_value.item():.4f}")

            # Backward and step
            t8 = time.time()
            combined_loss = loss_policy + value_loss_weight * loss_value
            combined_loss.backward()
            optimizer.step()
            t9 = time.time()
            # print(f"Batch {batch_idx}: Backward+Step time: {t9 - t8:.4f}s | combined_loss={combined_loss.item():.4f}")

            total_loss += combined_loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

            end_batch = time.time()
            # print(f"Batch {batch_idx}: Total batch time: {end_batch - start_batch:.4f}s\n")

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        avg_policy_loss = total_policy_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        avg_value_loss = total_value_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        return avg_loss, avg_policy_loss, avg_value_loss

    # def train_epoch(self,
    #             dataloader: DataLoader, 
    #             optimizer: optim.Optimizer, 
    #             policy_criterion: nn.Module, 
    #             value_criterion: nn.Module, 
    #             device: torch.device, 
    #             value_loss_weight: float = 1.0) -> Tuple[float, float, float]:
    #     self.train()
    #     total_loss = total_policy_loss = total_value_loss = 0.0

    #     # Abilita il debug dell’autograd per tracciare esattamente l’operazione che genera nan
    #     torch.autograd.set_detect_anomaly(True)

    #     for batch_idx, (states, policy_targets, value_targets) in enumerate(dataloader):
    #         print(f"\n--- Batch {batch_idx} ---")
    #         # 1) Controlla i dati in ingresso
    #         print("states:", torch.isnan(states).any().item(),      # True se ci sono nan
    #             "policy_targets:", torch.isnan(policy_targets).any().item(),
    #             "value_targets:", torch.isnan(value_targets).any().item())

    #         states = states.to(device)
    #         policy_targets = policy_targets.to(device)
    #         value_targets = value_targets.to(device)

    #         optimizer.zero_grad()
    #         # 2) Forward pass
    #         policy_logits, value_preds = self(states)
    #         print("logits min/max:", policy_logits.min().item(), policy_logits.max().item())
            
    #         # 3) Log‐softmax
    #         log_probs = F.log_softmax(policy_logits, dim=1)
    #         print("log_probs min/max:", log_probs.min().item(), log_probs.max().item())

    #         # 4) Calcola le loss
    #         #    (adatta qui al criterio che stai usando)
    #         policy_targets_flat = policy_targets.view(log_probs.size(0), -1)
    #         loss_policy = -(policy_targets_flat * log_probs).sum(dim=1).mean()
    #         loss_value = value_criterion(value_preds, value_targets.unsqueeze(1).float())
    #         print("loss_policy:", loss_policy.item(), "loss_value:", loss_value.item())

    #         combined_loss = loss_policy + value_loss_weight * loss_value
            
    #         # 5) Se già qui è nan, possiamo alzare un’eccezione o break
    #         if torch.isnan(combined_loss).any():
    #             raise RuntimeError(f"combined_loss è NaN al batch {batch_idx}")

    #         # 6) Backward + gradient clipping
    #         combined_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    #         optimizer.step()

    #         total_loss += combined_loss.item()
    #         total_policy_loss += loss_policy.item()
    #         total_value_loss += loss_value.item()

    #         # 7) Stampa intermedia
    #         if batch_idx % 50 == 0:
    #             print(f"→ Dopo ott {batch_idx}: Loss comb.: {combined_loss.item():.4f}")

    #     # Resto dei calcoli per le medie...
    #     avg_loss = total_loss / len(dataloader)
    #     avg_policy_loss = total_policy_loss / len(dataloader)
    #     avg_value_loss = total_value_loss / len(dataloader)
    #     return avg_loss, avg_policy_loss, avg_value_loss


    def train_network(self, 
                    # train_data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                    ts: str,
                    iteration: int,
                    num_epochs: int = 10, 
                    batch_size: int = 64, 
                    learning_rate: float = 0.001, 
                    weight_decay: float = 1e-4, 
                    value_loss_weight: float = 1.0,) -> None:
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Addestramento su dispositivo: {device}")
        self.device = device
        self.to(device)

        # states_np, policy_targets_np, value_targets_np = train_data
        
        # states_tensor: torch.Tensor = torch.tensor(states_np, dtype=torch.float32)
        # policy_targets_tensor: torch.Tensor = torch.tensor(policy_targets_np, dtype=torch.float32) 
        # value_targets_tensor: torch.Tensor = torch.tensor(value_targets_np, dtype=torch.float32)

        # train_dataset: TensorDataset = TensorDataset(states_tensor, policy_targets_tensor, value_targets_tensor)
        # train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True if len(train_dataset) > batch_size else False)

        dataset = ConvDataset(folder_path=f"data/{ts}/iteration_{iteration}")
        #dataset = NpzDataset(folder_path=f"data/{ts}/iteration_{iteration}")
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


        # policy_criterion: nn.Module = nn.CrossEntropyLoss() 
        policy_criterion = nn.KLDivLoss(reduction="batchmean")
        value_criterion: nn.Module = nn.MSELoss()
        optimizer: optim.Optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print(f"Inizio addestramento per {num_epochs} epoche...")
        for epoch in range(num_epochs):
            print(f"\nEpoca {epoch + 1}/{num_epochs}")
            if len(train_dataloader) == 0:
                print(" DataLoader è vuoto, impossibile addestrare l'epoca.")
                continue
            avg_loss, avg_policy_loss, avg_value_loss = self.train_epoch(
                train_dataloader, optimizer, policy_criterion, value_criterion, device, value_loss_weight
            )
            print(f"Fine Epoca {epoch + 1}: "
                f"Loss Media: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})")
            
            # Checkpoint del modello
            # torch.save(model.state_dict(), f"hive_model_epoch_{epoch+1}.pth")

        print("\nAddestramento completato.")
        # Salva il modello finale
        # torch.save(model.state_dict(), "hive_model_final.pth")
        # print("Modello finale salvato come hive_model_final.pth")

    def predict(self, T_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Previsione della rete neurale.
        T_in: Matrice di input
        Returns:
            policy_logits: Logits della policy (distribuzione di probabilità)
            value_output: Valore previsto per lo stato
        """
        self.eval()
        with torch.no_grad():
            T_in_tensor: torch.Tensor = torch.tensor(T_in, dtype=torch.float32).to(self.device)
            policy_logits, value_output = self(T_in_tensor)
            return value_output.cpu().numpy(), policy_logits.cpu().numpy()
            


if __name__ == '__main__':
    hive_net: NeuralNetwork = NeuralNetwork()

    num_samples: int = 256 
    
    # Stati: (N, C, H, W) -> (N, 196, 56, 56)
    dummy_states: np.ndarray = np.random.rand(num_samples, INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    # Target della Policy: (N, POLICY_OUTPUT_SIZE) -> (N, 615424)
    # Devono essere distribuzioni di probabilità valide (ogni riga somma a 1)
    dummy_policy_logits_targets: np.ndarray = np.random.rand(num_samples, POLICY_OUTPUT_SIZE).astype(np.float32)
    exp_logits: np.ndarray = np.exp(dummy_policy_logits_targets - np.max(dummy_policy_logits_targets, axis=1, keepdims=True))
    dummy_policy_targets: np.ndarray = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Target del Valore: (N,) -> valori scalari tra -1 e 1
    dummy_value_targets: np.ndarray = np.random.uniform(-1, 1, num_samples).astype(np.float32)
    
    dummy_train_data: Tuple[np.ndarray, np.ndarray, np.ndarray] = (dummy_states, dummy_policy_targets, dummy_value_targets)

    hive_net.train_network(
        dummy_train_data, 
        num_epochs=2, 
        batch_size=32, 
        learning_rate=0.001,
        value_loss_weight=0.5 
    )

    print("\nEsempio di addestramento con dati fittizi completato.")