Scrivi in Python, utilizzando PyTorch Geometric, una classe `HiveGNN` che implementi una Graph Neural Network per il gioco Hive con le specifiche seguenti:

**1. Input**

* Un grafo non diretto `G` con N nodi, dove ogni nodo rappresenta una cella occupata da un pezzo o una posizione vuota. Le caratteristiche di ciascun nodo `i` sono:

  * `player_color`: vettore di lunghezza 2, codifica binaria con:

    * `[0, 1]` se il pezzo è White
    * `[0, 0]` se il pezzo è Black
    * `[1, 0]` se il nodo è vuoto (nessun pezzo)
  * `insect_type`: vettore one-hot di lunghezza 8 corrispondente ai tipi `['Q', 'S', 'B', 'G', 'A', 'M', 'L', 'P']`; tutti zeri se il nodo è vuoto
  * `pinned`: valore binario 1 se c'è un beetle sopra, 0 altrimenti
  * `is_articulation`: valore binario 1 se il nodo è un articulation point del grafo di adiacenza (regola one-hive), 0 altrimenti

* Una matrice di adiacenza direzionale separata `move_adj` (dimensione N×N) che rappresenta per ogni nodo i possibili target di mosse valide: `move_adj[i, j] = 1` se esiste una mossa valida dal nodo `i` verso il nodo `j`, altrimenti 0. Questa matrice viene fornita solo per calcolare la policy `pi`, **non** viene usata nei message-passing per aggiornare gli embedding dei nodi.

**2. Architettura**

* Usa due moduli di message passing (ad esempio `GCNConv` o `GATConv`):

  1. **Adjacency Message-Passing**: applica K layer di GNN sul grafo non diretto `G` usando solo i suoi edge di adiacenza, per ottenere un embedding `h_i` per ciascun nodo `i`.
  2. **Action Scoring**: definisci un piccolo MLP che, preso in input la concatenazione `[h_i || h_j]` per ogni coppia `(i, j)` tale che `move_adj[i, j] = 1`, produce uno score `s_{i,j}`. Applica softmax su tutti questi score per ottenere la distribuzione di probabilità `pi` sulle mosse valide.

* **Value Head**: dopo l’ultima layer di adjacency message-passing, aggrega gli embedding dei nodi (ad esempio con mean-pooling su tutti `h_i`) e passa il risultato a un MLP che produce uno scalare `V` (stima della value function della board, tra -1 e 1).

**3. Output**

* `pi`: un vettore di dimensione M (numero di mosse valide), con `pi_{k} = exp(s_{i_k,j_k}) / sum_{all moves} exp(s_{i,j})` per ciascuna mossa `k` corrispondente a una entry `move_adj[i_k,j_k] = 1`.
* `V`: valore scalare (float) in \[-1, 1].

**4. Dettagli di implementazione**

* Definisci una classe `HiveGNN(torch.nn.Module)` con:

  * `__init__`: inizializza i layer di GNN, gli MLP per policy e value head.
  * `forward(data, move_adj)`: dove `data` è un oggetto `Data(x, edge_index)` di PyTorch Geometric e `move_adj` è la matrice N×N in formato torch.Tensor.
  * Rendi compatibile il forward per batch di grafi diversi (usa `batch` se necessario)

* Includi commenti chiari per ogni blocco e descrivi le dimensioni di input/output dei layer.

* Prevedi una loss combinata:

  ```python
  loss = F.cross_entropy(pi_logits, pi_target) + c_v * F.mse_loss(V_pred, v_target)
  ```

  dove `pi_target` e `v_target` provengono dal dataset generato in precedenza.

**5. DataLoader e training**

* Implementa una classe `HiveDataset(torch.utils.data.Dataset)` che:

  * Nel costruttore prende una lista di path directory.
  * Per ogni `game_x.json` (lista di `(pi_list, value)` per ogni mossa) e corrispondente `game_x.txt` (board finale in formato testuale), carica:

    1. `pi_list`: lista di coppie `(move_str, prob)` per ogni mossa.
    2. `value_list`: lista di float corrispondenti.
    3. Legge la board finale da `game_x.txt`, ricostruisce grafo `Data(x, edge_index)`, calcola `move_adj` e targets:

       * `pi_target`: tensor con probabilità corrispondenti all’ordine delle mosse valide.
       * `v_target`: tensor dei value scalari.
  * `__getitem__` restituisce `(data, move_adj, pi_target, v_target)`.

* Implementa un `DataLoader` PyTorch standard su `HiveDataset` con batching personalizzato che unisce liste di move\_adj variabili.

* Fornisci una funzione `train_epoch(model, dataloader, optimizer, device)` che:

  * Itera su batch, sposta dati su `device`.
  * Calcola `pi_pred, v_pred = model(data, move_adj)`.
  * Calcola `loss = F.cross_entropy(pi_pred_logits, pi_target) + c_v * F.mse_loss(v_pred, v_target)`.
  * Esegue backprop e optimizer step.
  * Ritorna `avg_loss` per epoch.

* Fornisci un wrapper `train_loop(model, dataset_paths, epochs, lr, device)` che:

  * Crea `HiveDataset` e `DataLoader`.
  * Init optimizer e learning rate scheduler.
  * Esegue `train_epoch` per ogni epoca, loggando loss e salvando checkpoint del modello.

**6. Esempio di utilizzo**

```python
# Dataset e DataLoader
dataset = HiveDataset(paths=["data/2025-07-12_10-00-00/pro_matches/"])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=hive_collate)

# Modello, device, training
model = HiveGNN(num_node_features=12, hidden_dim=128, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loop di training
train_loop(model, ["data/2025-07-12_10-00-00/pro_matches/"], epochs=50, lr=1e-3, device=device)
```


x = [0,1,  0,1,0,0,0,0,0,0,   0, 1]

Data ()

mat_adj = [
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
]
(simmetrica)

28 + nodi vuoti

mat_moves = [
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
]
(non simmetrica)