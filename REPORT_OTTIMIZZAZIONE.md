# Bees_Knees — Report bug e ottimizzazioni (branch `mcts-gat`)

Scan completo di `src/engine`, `src/ai`, `src/engineer.py`, `src/test`, `src/train_gnn.py`,
`src/test_mcts.py`. Obiettivo: massimizzare il numero di rollout MCTS al secondo.

Le misure sono prese su macOS/Python 3.13 (miniconda), senza torch, su una posizione a 20 ply
con 25 mosse legali. I numeri servono come ordine di grandezza, non come benchmark assoluto.

---

## 0. TL;DR — le cinque cose che contano

| # | Problema | Dove | Impatto |
|---|---|---|---|
| 1 | Tabella Zobrist per-istanza: 4.128.768 interi random per ogni `Board()` | `engine/hash.py:16` | **2.9 s e 237 MB per ogni `Board()`**; rende `deepcopy(board)` e `board.copy()` inutilizzabili |
| 2 | `MCTS_BATCH` chiama `self.init_board.copy()`, metodo che **non esiste** | `ai/mcts_batch.py:295,364` | `AttributeError` immediato: il searcher batched è **non funzionante** |
| 3 | `configure_optimizers()` dentro il loop dei batch: nuovo AdamW ad ogni step | `ai/graph_network.py:357` | Stato di Adam azzerato ad ogni batch → **il pretraining della GNN è di fatto rotto** |
| 4 | `count_queen_neighbors` ricalcolato 2× per ogni `safe_play` | `engine/board.py:106-107` | **~40% del tempo di `safe_play`+`undo`** |
| 5 | `stringify_move()` eseguito ad ogni `safe_play`, anche quando la stringa non serve | `engine/board.py:86-87` | **~17% del tempo di `safe_play`+`undo`** |

Costo attuale misurato di un `safe_play` + `undo`: **13.2 µs**. Con gli interventi 4, 5 e le
ottimizzazioni della sezione 3 è ragionevole scendere sotto i 4 µs, cioè **~3× rollout in più**
a parità di tempo, prima ancora di toccare la rete.

---

## 1. Bug bloccanti

### 1.1 `Board` non ha `copy()` — `MCTS_BATCH` crasha
`ai/mcts_batch.py:364` e `:295` chiamano `self.init_board.copy()`. In `engine/board.py` non
esiste né `copy` né `__deepcopy__`. Ogni discesa dell'albero solleva `AttributeError`.
Nota: anche aggiungendo un `copy()` ingenuo, con la tabella Zobrist attuale (bug 1.2) una copia
costerebbe centinaia di MB. Vanno risolti insieme.

### 1.2 Tabella Zobrist per-istanza, 4.1 M interi
`engine/hash.py:16` costruisce `_hashPartByPosition` come lista annidata
`28 × 128 × 128 × 9 = 4.128.768` interi da 64 bit, **una per ogni `Board`**.

Misurato: `Board()` = **2.909 s**, **236.9 MB**.

Conseguenze a catena:
- `Engine.newgame()` in ogni partita di `duel`/`duel_random` paga 3 s e 237 MB.
- `deepcopy(current.board)` in `AlphaBetaPruner._ab_pruning` (`ai/brains.py:117`) copia anche la
  tabella → inutilizzabile.
- Due `Board` diversi hanno tabelle random **diverse**, quindi i loro zobrist non sono
  confrontabili: qualunque cache condivisa fra board è silenziosamente errata.

Fix: tabella unica a livello di classe, costruita una sola volta con seed fisso, e come array
piatto indicizzato aritmeticamente. Ridurre anche il dominio: Hive non esce da ~±16 caselle, quindi
`_BOARD_SIZE = 40` e `_BOARD_STACK_SIZE = 7` bastano → 28 × 40 × 40 × 8 = 358 k valori, in un solo
`array('q')` o `np.ndarray` (≈3 MB, costruzione in millisecondi).

### 1.3 Ottimizzatore ricreato ad ogni batch
```python
# ai/graph_network.py:350-362
for batch in tqdm(train_loader, ...):
    ...
    optimizer = self.configure_optimizers()   # <-- dentro il loop
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```
AdamW viene istanziato da zero ad ogni batch: momenti primo/secondo ordine sempre a zero, nessun
bias correction sensato, warm-up dello scheduler inesistente. È il motivo più probabile di
convergenza scadente. In più `train_network:381` crea un ottimizzatore che non viene mai usato.

Fix: creare l'ottimizzatore una volta in `train_network` e passarlo a `train_epoch`.

### 1.4 `AlphaBetaPruner` chiama `get_valid_moves()` con un argomento
`ai/brains.py:153`: `board.get_valid_moves(PlayerColor.WHITE)` ma la firma è
`get_valid_moves(self)` → `TypeError` appena si valuta un nodo `IN_PROGRESS`. L'alpha-beta è morto.

### 1.5 `torch.autocast(dtype=...)` senza `device_type`
`ai/oracleGNN.py:147`: `with torch.autocast(dtype=amp_dtype):`. La firma è
`torch.autocast(device_type, dtype=None, ...)`, `device_type` è posizionale obbligatorio →
`TypeError` sul ramo CUDA di `predict_values_batch_from_data`, che è **il path usato da
MCTS_BATCH su GPU**. Da verificare sulla macchina CUDA, ma la correzione è comunque
`torch.autocast("cuda", dtype=amp_dtype)`.
La variante `..._with_gpu` (`:174`) usa `torch.cuda.amp.autocast`, deprecata ma funzionante.

### 1.6 `except:` nudo nel batch processor
`ai/mcts_batch.py:221`. Serve a intercettare il timeout di `Queue.get`, ma cattura **qualunque**
eccezione sollevata dal blocco di processing sottostante (mapping dei risultati, expand,
backprop). Un bug nel batch diventa un rollout silenziosamente perso invece di un crash.
Fix: `except Empty: pass` (`from queue import Empty`), e il resto del corpo fuori dal `try`.

### 1.7 Rollout persi a fine ricerca
Sempre nel batch processor: la condizione di flush è valutata **solo dopo un `get()` riuscito**.
Quando i worker terminano e restano in `pending` meno elementi della soglia, il `get` va in
timeout, il loop esce e **quei leaf non vengono mai valutati né backpropagati**. Va aggiunto un
flush finale fuori dal `while`.

---

## 2. Bug di correttezza

### 2.1 `_draw_counter` va in deriva sulle mosse `pass`
`engine/board.py:115` incrementa `_draw_counter[zobrist]` **dentro** `if move:`, mentre
`undo()` (`:135`) lo decrementa **incondizionatamente**. Ogni `pass` annullato decrementa un
contatore mai incrementato. Dopo abbastanza pass/undo i contatori vanno negativi e la regola
delle tre ripetizioni smette di scattare. Fix: spostare la riga 135 dentro `if move:`.

### 2.2 La cache delle mosse legali è indicizzata solo sullo zobrist
`get_valid_moves()` memorizza in `self._snapshots[self.zobrist_key]`, ma il set di mosse legali
dipende anche da `self.turn`: turno 0/1 hanno regole speciali e `current_player_turn != 4`
governa l'obbligo della regina. Lo zobrist codifica solo la **parità** del turno, non il numero.
Due posizioni identiche raggiunte a turno 4 e a turno 20 condividono la chiave → si può servire
un set di mosse calcolato con la regola della regina attiva quando non lo è più (o viceversa).
Fix: chiave `(zobrist_key, turn)` per i primi ~10 ply, o semplicemente includere `turn` nella
chiave finché `current_player_turn <= 4`.

### 2.3 `reward()` può restituire valori fuori da [0,1]
`ai/node_mcts.py:100-118`:
- ramo `DRAW`: `return 1 - self.V` con `self.V` inizializzato a `-1` → restituisce **2** se il
  nodo non è mai stato valutato. La riga `return 0.5` sotto è codice morto.
- ramo `else`: `return self.V`, cioè **-1** per un nodo non espanso.
Entrambi finiscono direttamente in `W` e `Q` in backprop e corrompono la selezione UCT.

### 2.4 Argomenti di default valutati sulla `property`
`engine/board.py:212` `def count_queen_neighbors(self, color: PlayerColor = current_player_color)`
e `:306` `_get_valid_placements(self, color = current_player_color)`. Nel corpo della classe
`current_player_color` è l'oggetto `property`, non un colore. Una chiamata senza argomento
costruisce `Bug(<property>, QUEEN_BEE)`, `_pos_from_bug` restituisce `None` e la funzione
ritorna **0 in silenzio**. Oggi tutte le chiamate passano il colore, ma è una mina.

### 2.5 `use_layer_norm` ignorato quando `use_residual=True`
`ai/graph_network.py:77`: `ResidualGNNBlock` istanzia la norm solo se `use_batch_norm`, altrimenti
`Identity()`. La configurazione usata in `test_mcts.py` (`use_layer_norm: True`,
`use_residual: True`) gira quindi **senza normalizzazione**, contrariamente all'intento.

### 2.6 `self.num_rollouts` sovrascritto dal risultato
`ai/mcts_batch.py:352`: `self.num_rollouts = completed_rollouts`. La configurazione viene
distrutta: alla mossa successiva il budget è quello effettivamente raggiunto la volta prima, e
degrada monotonicamente. Stesso difetto in `brains.MCTS` col ramo a tempo (`:331`).
Usare un attributo separato `self.last_rollouts`.

### 2.7 `_uct_select` con `node.N == 0`
`ai/brains.py:310`: `math.sqrt(node.N-1)` → `ValueError: math domain error` se `N == 0`.
`mcts_batch` usa già `math.sqrt(max(1, node.N))`; allineare.

### 2.8 Tree reuse fragile in `MCTS_BATCH`
`run_simulation_from` cerca fra i figli di `init_node` quello con lo zobrist corrente e
altrimenti fa `raise Error("No matching child found")`. Qualsiasi divergenza (mossa
dell'avversario fuori dall'albero, collisione zobrist, riuso del brain su una partita nuova)
termina la ricerca con un'eccezione invece di ricostruire l'albero. Il fallback esiste già
commentato nel backup.

### 2.9 Statistiche/threading
- `completed_rollouts` viene incrementato dal batch processor ma i worker lo leggono per decidere
  quando fermarsi: superano il target dell'intera coda in volo.
- `actual_threads = min(num_threads, max(1, self.num_rollouts // 10))` usa `num_rollouts` che nel
  ramo a tempo è il valore stantio del costruttore.
- `_backpropagate_remove_virtual` non aggiorna `Q` quando rimuove la virtual loss (solo `N`/`W`);
  è corretto solo perché i nodi coinvolti sono tutti antenati del leaf e vengono riattraversati.
  Resta fragile.
- La virtual loss non viene mai applicata al leaf finale, quindi due thread possono collidere
  esattamente sullo stesso leaf — proprio il caso che la virtual loss dovrebbe evitare.

### 2.10 Minori
- `engine/board.py:193` `for d in Direction` nel DFS di Tarjan include `ABOVE`/`BELOW`, che mappano
  entrambi su `(0,0)`: due self-loop inutili per nodo. Stesso problema in `_get_sliding_moves:335`.
- `oracleGNN._data_from_board`: `num_features = 13` ma gli indici usati sono `0`, `2..9`, `10..12`.
  **La colonna 1 non viene mai scritta**: un feature su 13 è costantemente zero (off-by-one su
  `type_to_index`, che parte da 1 e viene sommato a 1).
- `ai/loader.py`: `GraphDataset` usa il **primo** `.pkl` trovato nella cartella come cache, senza
  verificare che corrisponda ai dati; una cache stantia viene caricata in silenzio.
- `test_mcts.py:67-68` istanzia e carica il modello **a import time**, con path relativo
  `../models/...` dipendente dalla cwd.
- `ai/oracleGNN.py:33` `self.cache` dichiarata e mai usata.
- `ai/oracleGNN.py:104` `copy()` solleva se `self.path` è `None`, ed è marcata "probabilmente
  sbagliata" nel codice.
- `gen_dataset/{data_generator,match_generator,match_gen_parallel,graph_db_converter}.py`
  importano `ai.training`, modulo cancellato: quattro script già morti.

---

## 3. Ottimizzazioni del motore di gioco (il collo di bottiglia vero)

Profilo di 10.000 cicli `safe_play`+`undo` (1.130 s totali, cProfile):

| funzione | cumtime | quota |
|---|---|---|
| `count_queen_neighbors` (+ genexpr) | 0.452 s | **40%** |
| `stringify_move` | 0.190 s | 17% |
| `Position.get_neighbor` (+ `__add__`) | 0.177 s | 16% |
| `Bug.__str__` (via `BugName[...]` e `PlayerColor.code`) | 0.131 s | 12% |
| `_bugs_from_pos` | 0.151 s | 13% |
| `_update_cut_pos` | 0.045 s | 4% (cache calda) |

Micro-benchmark:

| operazione | costo |
|---|---|
| `safe_play` + `undo` | 13.23 µs |
| idem con `stringify_move` stubbato | 10.78 µs |
| `hash(bug)` (= `hash(str(bug))`) | 0.28 µs |
| `hash(move)` | 0.36 µs |
| `Direction.flat()` | 0.11 µs |
| `BugName[str(bug)].value` | 0.30 µs |
| `stringify_move` | 2.20 µs |
| `get_valid_moves()` (cache calda) | 0.11 µs |
| `get_valid_moves()` a freddo | 0.30 ms |

### O1 — Non stringificare le mosse durante la ricerca *(impatto alto, sforzo basso)*
`safe_play` (`board.py:86-87`) calcola `stringify_move(move)` e fa `append` su `move_strings`
anche quando il chiamante è MCTS, che la stringa non la guarda mai. Costo: 2.2 µs su 13.2.
Fix: rendere la stringa lazy (`move_strings.append(None)` e calcolarla solo su richiesta in
`__str__`/UHP), oppure aggiungere un flag `record_strings=False` sul path di ricerca.

### O2 — Contatori incrementali di vicini della regina *(impatto alto)*
Oggi ogni `safe_play` ricalcola da zero i 6 vicini di entrambe le regine. Solo le mosse che
toccano una casella adiacente a una regina possono cambiare quei conteggi.
Fix: mantenere `self._queen_neighbors = {WHITE: n, BLACK: n}` aggiornato in modo incrementale in
`safe_play`/`undo` guardando solo `move.origin` e `move.destination`; da 12 lookup a ≤2 confronti.
Da solo vale circa un terzo del tempo di gioco/annullamento.

### O3 — Tabella dei vicini precalcolata *(impatto alto, sforzo basso)*
`Position.get_neighbor` fa un `match` su 8 casi, una lookup in `POSITIONS` e un `__add__` che fa
un'altra lookup. Le `Position` sono già internate in `Position.POSITIONS`: basta calcolare una
volta sola, al momento dell'internamento, `pos.neigh = (p0..p5)` e sostituire le chiamate con un
indice. `Direction.delta_index` esiste già ed è la chiave naturale.
Elimina ~150 k chiamate su 10 k play/undo.

### O4 — `Direction.flat()` come costante di modulo *(impatto medio, sforzo minimo)*
`enums.py:180` ricostruisce una lista di 6 elementi ad ogni chiamata, e le chiamate sono ovunque
nei loop caldi (`count_queen_neighbors`, `_get_valid_placements`, `_get_beetle_moves`,
`_get_grasshopper_moves`, `_data_from_board`, …).
Fix: `FLAT_DIRECTIONS = (…)` a livello di modulo. Stesso discorso per `opposite`, `left_of`,
`right_of`, `delta_index`, `is_left`, `is_right`: oggi sono `property` con `match` a catena,
vanno precomputati in tuple/dict indicizzati.
Anche `PlayerColor.code` (`enums.py:79`) fa `self[0].lower()` — slicing di stringa + `lower()` —
ad ogni chiamata: 128 k chiamate nel profilo. Va cachata.

### O5 — `Bug.__hash__` non deve costruire una stringa *(impatto medio-alto)*
`game.py:151`: `return hash(str(self))`. I `Bug` sono chiavi di `_bug_to_pos` e componenti di
`Move.__hash__`, e i `Move` vivono in `set` costruiti a ogni `get_valid_moves` a freddo.
Fix: assegnare in `__init__` un id intero denso (0..27, lo stesso di `BugName`) e usare
`self._hash = idx`. Stessa cosa per `Move.__hash__`, che può diventare
`(bug_idx << 40) ^ (origin_hash << 20) ^ dest_hash` senza costruire tuple.

### O6 — Eliminare `BugName[str(move.bug)].value` dalla hot path *(impatto medio)*
`board.py:82,97,101,102,139,143,151`: tre/quattro conversioni stringa + lookup enum per ogni
`safe_play`. Se `Bug` porta già il proprio indice (O5), diventa `move.bug.idx`.

### O7 — Limitare la crescita delle strutture ausiliarie *(impatto memoria, medio)*
Tutte per-`Board` e mai potate:
- `_snapshots: dict[int, set[Move]]` — un set di mosse per ogni posizione mai visitata.
- `_snapshots_art_pos: dict[int, set[Position]]`
- `_draw_counter: defaultdict`
- `_pos_to_bug` conserva le liste vuote delle caselle abbandonate, e `_update_cut_pos:182` le
  riscansiona tutte ad ogni cache miss.
Su una ricerca lunga questo cresce senza limite e peggiora progressivamente la località.
Fix: cache LRU con cap (es. 200 k voci), e `del self._pos_to_bug[pos]` quando la lista si svuota.

### O8 — `Random` non deve stringificare tutte le mosse
`ai/brains.py:53`: `board.valid_moves.split(";")` costruisce la stringa UHP di **tutte** le mosse
(2.2 µs × numero di mosse) per poi sceglierne una. In `duel_random` questo è metà delle mosse
della partita. Fix: `choice(tuple(board.get_valid_moves()))` e stringificare solo la scelta.

### O9 — Togliere assert e logging dal loop dei rollout
- `ai/brains.py:309` `assert node.N-1 == sum(child.N for child in node.children)`: una somma
  O(figli) **ad ogni passo di selezione**.
- `ai/node_mcts.py:52`: altro `assert` con somma sui figli.
- `ai/brains.py:258,269,299`: `print_log(f"...")` — la f-string viene costruita anche se
  `print_log` ritorna subito.
- `ai/brains.py:311` definisce la closure `uct_Norels` ad ogni chiamata di `_uct_select`.
- `ai/mcts_batch.py:110,113,117`: `print("TREE DA CREARE")` ecc. ad ogni mossa.
- `ai/log_utils.py:26` `countit` decora `_collect_leaf_for_batch` nel backup: un wrapper
  `functools.wraps` con `try/finally` per chiamata.

### O10 — Tarjan incrementale
`_update_cut_pos` rifà da zero il calcolo dei punti di articolazione ad ogni cache miss, con DFS
**ricorsiva** (rischio `RecursionError` su hive lunghe). Due miglioramenti indipendenti:
riscrivere il DFS in forma iterativa, e valutare un algoritmo incrementale che ricalcoli solo la
componente toccata da `move.origin`/`move.destination`.

---

## 4. Ottimizzazioni lato rete / torch

### N1 — Controllare la cache **prima** di costruire il `Data` *(impatto altissimo)*
In `_collect_leaf_with_virtual_loss` (`mcts_batch.py:390-410`) per ogni mossa legale si esegue
`self.oracle._data_from_board(board_copy)` e solo dopo, nel batch processor, si verifica se lo
zobrist è già in `hashmap`. Il valore cachato evita la forward della rete ma **non** evita la
costruzione del grafo, che è la parte Python più costosa.
Fix: calcolare `state_hash`, consultare `hashmap` nel worker, e costruire il `Data` solo in caso
di miss. Su posizioni con molte trasposizioni è il singolo guadagno più grosso di tutto il file.

### N2 — La cache V viene svuotata ad ogni mossa
`mcts_batch.py:102-103`: `self.hashmap.clear()` ad ogni `run_simulation_from`. Nel backup la riga
era **commentata**. Le valutazioni di rete della mossa precedente sono perfettamente riutilizzabili
(stesso modello, stesse posizioni). Fix: non svuotare, e mettere un cap LRU.

### N3 — Costanti ricostruite ad ogni `_data_from_board`
`oracleGNN.py:282-284`: `types = list(BugType)` e `type_to_index = {...}` vengono ricostruiti ad
ogni chiamata, cioè una volta per mossa legale per leaf. Vanno a livello di modulo.

### N4 — `pos_bug_to_index` costruito e mai usato
`oracleGNN.py:290,309`: un dict con chiavi `(Position, Bug)` — quindi con hashing di `Bug`, cioè
costruzione di stringa (vedi O5) — popolato per ogni nodo e **mai letto**. Da cancellare.

### N5 — `pin_memory()` per ogni micro-tensore
`oracleGNN.py:343-345`: `pin_memory()` su tre tensori di ~30 righe, per ogni `Data`.
`pin_memory` è una allocazione host page-locked con sincronizzazione: su tensori così piccoli il
costo supera abbondantemente il beneficio del trasferimento asincrono.
Fix: pinnare, se serve, **una sola volta** il `Batch` aggregato in
`predict_values_batch_from_data`, non i singoli `Data`.

### N6 — Costruzione delle edge come lista di tuple Python
`oracleGNN.py:314-337`: `edge_list` è una lista di tuple, convertita con
`torch.tensor(edge_list).t().contiguous()` — path lento di PyTorch.
Fix: riempire un `np.empty((2, E), dtype=np.int64)` e usare `torch.from_numpy`.
Inoltre il raggruppamento `by_height` ricostruisce un `defaultdict(dict)` ad ogni chiamata: per
Hive l'altezza massima è ~7 e quasi tutti i nodi sono a h=0, conviene un `list` indicizzato.

### N7 — `torch.compile(mode="reduce-overhead")` è probabilmente controproducente qui
`oracleGNN.py:43-47`. Due problemi:
1. `torch.compile` è lazy: la compilazione avviene alla prima forward, quindi il `try/except`
   intorno alla chiamata **non intercetta nulla**.
2. `reduce-overhead` attiva i CUDA graph, che richiedono shape stabili. Le batch di MCTS hanno un
   numero di nodi e di archi che cambia ad ogni chiamata → ricompilazioni continue o fallback,
   con overhead netto negativo.
Fix: misurare `mode="default"` con `dynamic=True` contro nessuna compilazione, e in ogni caso
spostare il `try/except` intorno a una forward di warm-up.

### N8 — Flag globali settati ad ogni chiamata
`oracleGNN.py:143-144` imposta `allow_tf32` su matmul e cudnn ad ogni
`predict_values_batch_from_data`. Vanno in `__init__`.

### N9 — `model.eval()` ad ogni `predict`
`graph_network.py:287`: `self.model.eval()` itera tutti i sottomoduli ad ogni chiamata. Con
migliaia di chiamate per mossa è overhead puro: mettere il modello in eval una volta
nel costruttore dell'oracle.

### N10 — DataLoader configurato al contrario
`oracleGNN.py:69-71`: il ramo **CUDA** usa `num_workers=0`, il ramo **CPU** usa
`num_workers=6, pin_memory=True, persistent_workers=True`. `pin_memory` senza GPU è overhead
inutile. (Il `num_workers=0` su CUDA è giustificato perché il dataset è preallocato su GPU, ma
va documentato.)

### N11 — Il threading non può accelerare la discesa dell'albero
`MCTS_BATCH` lancia `num_threads` worker Python che fanno discesa dell'albero, `safe_play`,
`undo` e costruzione dei `Data`: tutto codice Python puro, quindi **serializzato dal GIL**, con in
più `tree_lock` tenuto intorno all'intera selezione (`mcts_batch.py:157`) e tre lock globali.
Il risultato realistico è più lento del ciclo sequenziale del backup.
Le due strade sensate sono: (a) tornare al loop sequenziale batched del backup e investire sul
batch size e sulla cache; (b) parallelizzare a livello di **processi** (`multiprocessing`) con la
rete su un server di inferenza, che è anche il modello usato da AlphaZero.
Prerequisito per (b): una `Board` serializzabile e leggera, cioè il fix 1.2.

### N12 — Il ciclo `O(n²)` nel mapping dei risultati
`mcts_batch.py:245`: `[p for i, p in enumerate(pending) if i in leaf_batch_indices]` fa un test di
appartenenza su una **lista** dentro una comprehension → quadratico nella dimensione del batch.
Fix: `leaf_batch_indices` come `set`, o meglio salvare direttamente `(idx, item)` quando si
accoda.

### N13 — Espansione ripetuta dei nodi già espansi
Il nuovo `mcts_batch` ha rimosso il ramo `is_unexplored and is_expanded` presente nel backup.
Adesso un nodo già espanso ma "resettato" rifà la forward della rete per il leaf **e per tutti i
figli**, poi `expand()` scopre di essere già espanso e si limita a `reset_children()`, buttando
via tutto il lavoro. Va reintrodotto il corto-circuito.

---

## 5. Ottimizzazioni algoritmiche MCTS

- **`brains.MCTS` non riusa l'albero**: `run_simulation_from:320` crea un `Node_mcts` radice nuovo
  ad ogni mossa, buttando l'intero albero della mossa precedente. `MCTS_BATCH` il riuso ce l'ha
  (sezione 2.8). Portarlo anche nella versione sequenziale vale tipicamente 20-40% di rollout utili.
- **Nessuna tabella di trasposizione sui nodi**: due percorsi che arrivano alla stessa posizione
  creano due sottoalberi indipendenti. Con lo zobrist già disponibile, un dict
  `hash -> Node_mcts` condiviso permette di fondere le statistiche.
- **`brains.MCTS:325`**: `if len(self.init_node.children) == 1: return` è controllo morto —
  `children` è sempre vuoto subito dopo la costruzione del nodo.
- **Nessun progressive widening / cutoff**: ad ogni espansione si valuta la rete su **tutte** le
  mosse legali (25-100 in mediogioco). Valutare solo le prime *k* per un ordinamento euristico
  taglia linearmente il costo di espansione.
- **`get_moves_probs`** divide per `self.num_rollouts`, che dopo il bug 2.6 non è più il totale
  reale: le policy target per l'allenamento non sommano a 1. `mcts_batch` usa già la somma dei
  figli; allineare `brains.MCTS:362`.
- **Il loop a tempo chiama `time()` due volte per rollout** (`brains.py:333`); con rollout da
  poche decine di µs non è trascurabile. Controllare il tempo ogni N rollout.

---

## 6. Differenze `mcts_batch.py` vs `mcts_batch_bak.py`

Il backup **non è** una copia quasi identica: è la versione **sequenziale** precedente.
`mcts_batch.py` è una riscrittura con discesa parallela e virtual loss.

| Aspetto | `mcts_batch_bak.py` (sequenziale) | `mcts_batch.py` (attuale) |
|---|---|---|
| Esecuzione | loop singolo, `_collect_leaf_for_batch` | `ThreadPoolExecutor`, N worker + 1 batch processor, `Queue` |
| Board durante la discesa | **riusa `self.init_board`** con play/undo | `self.init_board.copy()` → **metodo inesistente** |
| Virtual loss | assente | `N += 3 / W -= 3`, rimossa in backprop |
| Lock | nessuno | `tree_lock`, `cache_lock`, `stats_lock` |
| Cache V fra mosse | `hashmap.clear()` **commentato** → cache persistente | `hashmap.clear()` attivo → cache buttata ogni mossa |
| Nodo già espanso | ramo dedicato che corto-circuita (`"SI GODE, L'AVEVAMO GIA"`) | ramo rimosso → lavoro rifatto |
| Fallback tree reuse | `raise` + codice di ricostruzione (commentato) | solo `raise Error("No matching child found")` |
| Backprop | tre varianti separate (`_backpropagate`, `_backpropagate_N`, `_backpropagate_non_N`) con N e W aggiornati in momenti distinti | una sola `_backpropagate_remove_virtual` |
| `@countit` | su `_collect_leaf_for_batch` | rimosso |
| Flush finale | `flush_batch()` esplicito dopo il loop | assente (bug 1.7) |
| Gestione eccezioni | nessuna | `except:` nudo (bug 1.6) |

**Difetti condivisi da entrambi**: `self.num_rollouts = completed_rollouts`, la comprehension
quadratica su `leaf_batch_indices`, la costruzione dei `Data` prima del controllo in cache.

**Conclusione operativa**: il backup è l'unica delle due versioni che possa girare. La strada più
rapida verso un searcher funzionante è ripartire da `mcts_batch_bak.py`, applicare i fix N1, N2,
N12 e 2.6, e rivalutare la parallelizzazione solo dopo aver risolto 1.2 (Board leggera) e in forma
multi-processo.

---

## 7. Pulizia e debito minore

- `ai/brains.py:7` importa `deepcopy` usato solo dall'alpha-beta rotto; `ai/node_mcts.py:5`
  importa `uniform` mai usato; `oracleGNN.py` importa `Move` e `GraphDataset` marginalmente.
- `engine/game.py:51-126`: 75 righe di `Position` commentato.
- `board.py:433-449`: vecchia implementazione di `_can_move_without_breaking_hive` commentata.
- `ai/log_utils.py:5` `reset_log` ha un `return` prima del corpo.
- `brains.py:158` `print_log` idem; `print_log2` invece apre e chiude il file ad ogni riga.
- `test/` contiene l'harness di duello/training, non test: nome fuorviante e conflitto potenziale
  con `pytest`. C'è anche `test/mcts.py` + `test/tictactoe.py`, un MCTS di riferimento scollegato.
- `oracleGNN.py:46` messaggio di errore da ripulire.
- Nessun `pyproject.toml`: gli import assoluti dipendono dalla cwd e più script fanno `os.chdir()`
  a import time (`train_gnn.py:16`).

---

## 8. Ordine di esecuzione suggerito

**Fase 1 — sbloccare** (poche ore, nessuna scelta di design)
1. Zobrist condivisa a livello di classe + dominio ridotto (1.2)
2. `Board.copy()`/`__deepcopy__` efficiente, ora che la tabella è condivisa (1.1)
3. Ottimizzatore fuori dal loop (1.3)
4. `torch.autocast("cuda", ...)` (1.5)
5. `except Empty` + flush finale (1.6, 1.7)
6. `self.num_rollouts` non più sovrascritto (2.6)

**Fase 2 — velocità del motore** (il grosso del guadagno, misurabile senza GPU)
7. `stringify_move` lazy (O1)
8. Contatori incrementali della regina (O2)
9. Tabella dei vicini + `Direction` precalcolata (O3, O4)
10. `Bug`/`Move` con hash intero (O5, O6)
11. Assert e logging fuori dai loop caldi (O9)

**Fase 3 — velocità della rete**
12. Cache consultata prima di costruire il `Data` (N1)
13. Cache V persistente fra le mosse (N2)
14. Costanti fuori da `_data_from_board`, via `pos_bug_to_index`, edge via numpy (N3, N4, N6)
15. Via il `pin_memory` per-`Data` (N5)
16. Benchmark di `torch.compile` (N7)

**Fase 4 — architettura**
17. Decidere fra sequenziale-batched e multi-processo (N11)
18. Tabella di trasposizione sui nodi, tree reuse anche in `brains.MCTS`

**Come misurare.** Serve un benchmark riproducibile prima di toccare qualsiasi cosa: prendere una
delle posizioni di `test_mcts.py`, fissare il seed, e misurare *rollout/secondo* con
`restriction="time"`. Senza quello nessuna di queste ottimizzazioni è verificabile.
Il profilo della sezione 3 si riproduce con `cProfile` su un loop di `safe_play`/`undo`,
e non richiede torch.
