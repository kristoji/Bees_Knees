# Training su giano.cs.unibo.it

Procedura completa, dal clone ai pesi. Riferimento: *Istruzioni tecniche d'uso del
cluster GPU DISI*, aggiornamento gennaio 2025, **più quello che ho misurato sul cluster
il 2026-09-23** — dove le due cose divergono vale la misura, segnalata come tale.

## Quello che vincola tutto

| | |
|---|---|
| Quota home | **400 MB** → il venv (diversi GB) **deve** stare in `/scratch.hpc`, non nella home |
| Cache pip | sta nella home → **sempre `--no-cache-dir`**, altrimenti satura la quota |
| `/scratch.hpc` | i file non acceduti da **40 giorni vengono cancellati** |
| Coda `rtx2080` | **8 nodi, tutti liberi** (default). 4 core, 44 GB RAM, RTX 2080 Ti — Turing `sm_75`, 11 GB VRAM, **no bf16** |
| Coda `l40` | **4 nodi, tutti occupati con coda dietro**. 8 core, 64 GB RAM, L40 — Ada `sm_89`, 48 GB VRAM, bf16 |
| Limite di tempo | **3 giorni** su entrambe |
| GPU | driver **535.261.03**, che supporta fino a CUDA 12.2 → le wheel `cu118` vanno bene; `cu128` no |
| `/scratch.hpc` sui nodi | **visibile**, verificato con un job: `--chdir=/scratch.hpc/...` funziona |
| Python | **3.11.2**; `pip` di sistema **non esiste**, ma `python3 -m venv` si porta il suo (23.0.1) |
| Estrattori RAR | **nessuno**: niente `unrar`, `unar`, `7z`, `bsdtar`, né il modulo `rarfile` |
| `--gres=gpu:1` | va lasciata invariata: un nodo, una GPU |
| Submit | solo da `giano.cs.unibo.it`, con `sbatch` |

Il `requirements.txt` del repo **non è usabile qui**: pinna `torch==2.8.0+cu128` con le
estensioni PyG buildate per `pt24cu124`, incoerenti fra loro e con CUDA 11.8. Si usa
`cluster/requirements-giano.txt`, che installa torch dall'indice `cu118` e **non**
installa `torch_scatter/torch_sparse/torch_cluster/torch_spline_conv`: tutto ciò che
`graph_network.py` usa (GIN/GAT/GCN e i global pool) gira sugli scatter nativi di torch
in PyG ≥ 2.3, e quelle estensioni richiederebbero una build da sorgente lunga e fragile.

## 0. Ricognizione (da fare per prima)

```bash
ssh francesco.giordani5@giano.cs.unibo.it
mkdir -p /scratch.hpc/$USER
git clone -b mcts-gat https://github.com/kristoji/Bees_Knees.git /scratch.hpc/$USER/Bees_Knees
cd /scratch.hpc/$USER/Bees_Knees
bash cluster/discover.sh
```

Stampa nomi reali delle code, limiti di tempo, GPU, versione di Python e spazio. Serve a
confermare i parametri prima di pinnare qualunque cosa.

Il job da un minuto in fondo a `discover.sh` verifica che i nodi di calcolo vedano
`/scratch.hpc`. **Già verificato il 2026-09-23: lo vedono**, quindi la guida, che su questo
punto è ambigua, va letta nel senso che a non vederlo sono solo le macchine di laboratorio.

## 1. Ambiente

```bash
bash cluster/setup_giano.sh
```

Crea il venv in `/scratch.hpc/$USER/venv`, installa torch cu118 e il resto con
`--no-cache-dir`, e stampa una verifica. Su giano `torch.cuda.is_available()` sarà `False`:
è normale, la GPU ce l'hanno solo i nodi di calcolo.

## 2. Dataset

Il dataset originale serve **solo per i `board.txt`**: `x`, `edge_index` e `v` sono tutti
ricostruibili da lì (verificato: 100% delle etichette e, a parte la feature di
articolazione che era sbagliata, tutti i grafi). Quindi non servono i 10⁶ JSON.

**Su giano non c'è nessun estrattore RAR**, quindi l'archivio va aperto altrove. Sul Mac
(`bsdtar` legge RAR5) e poi rsync dei soli `board.txt`:

```bash
# in locale
bsdtar -xf Hive_dataset.rar -C raw --include='*/board.txt'      # ~70s, 82 MB
rsync -a raw/ francesco.giordani5@giano.cs.unibo.it:/scratch.hpc/francesco.giordani5/hive_raw/

# su giano
cd /scratch.hpc/$USER/Bees_Knees && sbatch cluster/rebuild.sbatch
```

Contenuto reale dell'archivio: **21049 partite** in 14 collezioni.

| gruppo | partite | incluso di default |
|---|---|---|
| `*_tournament` | 5162 | sì |
| `*_bots-after-2023` | 7714 | sì |
| `nokamute-*` | 2621 | sì — sono autopartite di motore, stessa categoria dei bot |
| `*_humans` | 3666 | **no** |

Gli shard sono per collezione, quindi la scelta si cambia senza rigenerare, con
`--collections` in `train_value.py`.

Produce shard `.npz` pre-collati in `/scratch.hpc/$USER/hive_shards` (~2 GB per l'intero
corpus) invece di un milione di file sciolti, che su un filesystem HPC sono un problema di
inode e di metadati. Sul campione di 259 partite il rebuild gira in 2,9 s, quindi l'intero
corpus è questione di minuti.

Controllo, anche in locale:

```bash
python tools/verify_dataset.py --shards /scratch.hpc/$USER/hive_shards
```

## 3. Training

```bash
sbatch cluster/train.sbatch                                  # l40, hidden 64, GIN x3
sbatch --partition=rtx2080 --mem=24G cluster/train.sbatch     # prova rapida
HIDDEN=128 LAYERS=4 EPOCHS=80 sbatch cluster/train.sbatch     # variante
```

Prima di lanciare la run vera, una di smoke:

```bash
EXTRA="--limit-games 200" EPOCHS=2 sbatch --time=00:10:00 cluster/train.sbatch
```

Output in `/scratch.hpc/$USER/models/value-<jobid>/`: `best.pt`, `train_log.jsonl`,
`summary.json`.

### Cosa guardare

Il log riporta a ogni epoca la BCE di validation **e** due baseline: il predittore
costante e l'euristica fatta a mano (conteggio dei vicini della regina). La domanda della
prima run non è "quanto è brava" ma **"batte l'euristica?"**. Se no, alla ricerca non
serve.

Lo split è **per partita**, non per posizione. È il motivo per cui l'overfitting si vede:
tutte le ~56 posizioni di una partita condividono la stessa etichetta, quindi uno split
per posizione mette esempi quasi identici su entrambi i lati e nasconde il problema.

Riferimento misurato in locale sulle sole 259 partite di campione (209 di training):
la rete scende a val BCE 0,614 contro 0,625 dell'euristica all'epoca 10, poi va in
overfitting netto e l'early stopping interviene. Con 15497 partite (tournament + bot +
nokamute) ci si aspetta molto meglio — ma è la ragione per cui si parte con
`--hidden-dim 64` e non 256.

## 4. Riportare i pesi e farli giocare

```bash
scp francesco.giordani5@giano.cs.unibo.it:/scratch.hpc/francesco.giordani5/models/value-*/best.pt models/
```

I pesi si caricano in `OracleGNN` con gli **stessi** iperparametri di architettura usati in
training (sono in `summary.json`):

```python
oracle = OracleGNN(device="cpu", hidden_dim=64, conv_type="GIN", num_layers=3,
                   use_layer_norm=True, use_residual=False, pooling="add",
                   mlp_layers=2, final_mlp_layers=2)
oracle.load("models/best.pt")
```

La misura che conta davvero non è la loss ma il **win rate di MCTS+GNN contro
MCTS+`Oracle` euristico a parità di rollout**, con `src/test/duel.py`.

## Problemi probabili

| sintomo | causa |
|---|---|
| `No space left` / `Disk quota exceeded` durante pip | la cache di pip nella home: usare `--no-cache-dir` e `pip cache purge` |
| `CUDA error: no kernel image is available` | wheel senza `sm_75`: sulla 2080 Ti serve una build che includa Turing. `torch==2.5.1+cu118` la include |
| job che non parte | `l40` ha quasi sempre coda; `rtx2080` di solito è libera |
| `best.pt` sparito dopo qualche settimana | `/scratch.hpc` cancella i file non acceduti da 40 giorni |
| la rete non batte l'euristica | vedi `TRAINING_and_DATASET.md` §4: poche etichette indipendenti, rete troppo grande, o troppe epoche |
