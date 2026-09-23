# Bees_Knees — Bug corretti e ottimizzazioni applicate (branch `mcts-gat`)

Aggiornato dopo l'esecuzione del piano. La prima metà del documento riporta cosa è
stato fatto e quanto è stato misurato; la seconda cosa resta aperto.

---

## 0. Risultato

A/B su 6 round alternati fra `bc7e37a` (prima delle ottimizzazioni) e `HEAD`:

| metrica | prima | dopo | fattore |
|---|---|---|---|
| `Board()` | 262 ms | 0,034 ms | **7745×** |
| `safe_play` + `undo` | 10,12 µs | 2,51 µs | **4,03×** |
| `get_valid_moves()` a freddo | 537 µs | 134 µs | **4,02×** |
| **MCTS rollout/s** | **1663** | **8443** | **5,08×** |

Riproducibile con `python bench/ab.py bc7e37a HEAD --rounds 6`.

Il motore è **comportamentalmente identico** al punto di partenza: `bench/diffcheck.py`
confronta il fingerprint di 1091 checkpoint (ogni set di mosse legali, stato e numero di
turno) su 10 posizioni iniziali, e `bc7e37a` e `HEAD` danno lo stesso digest
`1dcc1814…`. Le uniche differenze volute sono i bug di correttezza elencati sotto.

### Metodologia, e una correzione

Questa macchina è **bimodale di un fattore ~2,7** fra E-core e P-core: una singola
esecuzione può inventare o nascondere uno speedup per intero. Durante il lavoro ho
annunciato un "3,6× da Zobrist" che era **solo rumore di misura** — l'A/B interleaved
successivo ha mostrato 0,95×. Il messaggio del commit `a5b3d8e` contiene quel numero
sbagliato; vale questa tabella, non quella.

Da lì in poi ogni misura passa da `bench/ab.py`, che materializza due revisioni in due
directory e le alterna prendendo il best-of-N, così la deriva termica colpisce entrambe.

### Strumenti aggiunti

| file | cosa fa |
|---|---|
| `bench/bench_engine.py` | benchmark riproducibile, senza torch (usa l'`Oracle` euristico) |
| `bench/ab.py` | A/B interleaved fra due revisioni o il working tree (`.`) |
| `bench/difftest.py` | fingerprint deterministico del comportamento del motore |
| `bench/diffcheck.py` | confronta il fingerprint di due revisioni, exit code ≠ 0 se differisce |
| `bench/test_mcts_batch.py` | esercita `MCTS_BATCH` sul motore vero con torch stubbato |
| `bench/baseline.txt`, `bench/current.txt` | numeri prima/dopo |

---

## 1. Sui pesi della rete: il training va rifatto

Cercati su disco e in tutta la storia git. Risultato:

- **Nessun peso sul disco.** `models/`, `data/` e `logs/` sono gitignorati e assenti.
- **Un solo checkpoint è mai esistito nel repo**: `models/pretrain_0.pt` (359 KB),
  recuperabile da `git cat-file -p 3681f0e:models/pretrain_0.pt`.
- **Non è caricabile dal codice attuale.** Le sue chiavi sono
  `model.convs.N.conv.mlp.network.*` e `model.convs.N.conv.epsilon`: appartengono alla
  vecchia `GIN_Conv` scritta a mano, rimossa quando si è passati a PyG. La `GINConv` di
  PyG chiama la sua rete `nn` e il suo parametro `eps`, quindi le chiavi non
  corrispondono. È un GIN a 10 layer, hidden 64.
- I checkpoint a cui il codice fa riferimento — `models/gnn/pretrain_GIN_3.pt`,
  `models/GIN_epoch_30.pt`, `models/pretrain_GAT_5.pt` — **non esistono da nessuna parte**.

Quindi: **il pretraining va rifatto da zero**. Due conseguenze pratiche:

1. Serve anche il dataset di grafi, che non c'è: va rigenerato con `src/gen_dataset/`,
   dove però quattro script importano `ai.training`, modulo cancellato (vedi §4).
2. Poiché nessun peso esistente è riutilizzabile, il fix del layer norm (§2) e
   l'eventuale correzione della colonna di feature inutilizzata (§4) sono **gratis
   adesso** e non lo saranno più dopo il primo training serio.

---

## 2. Bug corretti

### Bloccanti

**Tabella Zobrist per istanza** (`engine/hash.py`) — 4.128.768 interi random in una lista
annidata a 4 livelli costruita in `__init__`: 262 ms e 237 MB **per ogni `Board()`**. E
ogni board aveva tabelle diverse, quindi le sue chiavi non erano confrontabili con quelle
di un'altra board: qualunque cache condivisa fra board era silenziosamente sbagliata.
Ora è una tupla piatta a livello di modulo, seed fisso, indicizzata con stride
precalcolati, e ogni `Position` porta il proprio offset. 262 ms → 0,034 ms.

**`MCTS_BATCH` non partiva** — chiamava `self.init_board.copy()` in due punti, metodo
inesistente: `AttributeError` alla prima discesa. Vedi §3.

**Ottimizzatore ricreato a ogni batch** (`ai/graph_network.py:357`) —
`configure_optimizers()` era **dentro** il loop dei batch, quindi AdamW ripartiva con
momenti primo e secondo a zero a ogni singolo step. Il pretraining girava di fatto senza
stato adattivo. Ora l'ottimizzatore è creato una volta in `train_network` e passato;
anche lo scheduler viene finalmente steppato.

**`torch.autocast(dtype=…)` senza `device_type`** (`ai/oracleGNN.py:147`) —
`device_type` è posizionale obbligatorio, quindi l'intero ramo CUDA della predict
batched sollevava `TypeError`: **il path usato da MCTS_BATCH su GPU era morto**.
Ora `torch.autocast("cuda", dtype=…)`.

**`AlphaBetaPruner` sollevava `TypeError`** — `board_evaluation` chiamava
`get_valid_moves(PlayerColor.WHITE)`, ma la funzione non prende argomenti. In più faceva
`deepcopy(board)` per ogni figlio, cioè copiava i 237 MB di tabella. Ora usa la mobilità
del giocatore di turno con segno, e `Board.copy()`.

**Crash sulla softmax vuota** (`ai/oracle.py`) — `np.max` su array vuoto quando il
giocatore può solo passare: MCTS moriva su qualunque posizione senza mosse legali. È il
primo bug che il benchmark ha trovato, prima ancora di misurare qualcosa.

### Correttezza

- **`_draw_counter` andava in deriva sui pass** — `safe_play` incrementava solo per le
  mosse vere, `undo` decrementava sempre. Dopo abbastanza pass annullati i contatori
  diventavano negativi e la regola delle tre ripetizioni non scattava più.
- **Cache delle mosse legali indicizzata solo sullo Zobrist** — ma le regole di apertura e
  l'obbligo della regina al quarto turno dipendono dal **numero** di turno, che la chiave
  codifica solo come parità. Ora la chiave è salata sui primi 8 ply.
- **`reward()` fuori da [0,1]** (`ai/node_mcts.py`) — restituiva `2` per un pareggio mai
  valutato (`1 - V` con `V = -1`) e `-1` per un nodo non terminale non espanso. Finiva
  dritto in `W` e `Q` lungo tutto il cammino.
- **Argomenti di default valutati sulla `property`** — `count_queen_neighbors` e
  `_get_valid_placements` avevano `color = current_player_color`, che nel corpo della
  classe è l'oggetto `property`. Una chiamata senza argomento restituiva 0 in silenzio.
- **`use_layer_norm` ignorato nei blocchi residuali** — ogni configurazione residuale
  girava **senza alcuna normalizzazione**, inclusa quella di `test_mcts.py` che la chiede
  esplicitamente. *Questo cambia l'albero dei moduli: i checkpoint precedenti non si
  caricherebbero comunque.*
- **`num_rollouts` sovrascritto dal risultato** — il budget decadeva di mossa in mossa; e
  `get_moves_probs` ci divideva, quindi le policy target non sommavano a 1.
- **`sqrt(N-1)` con `N = 0`** in `_uct_select`.
- **`Position`/`Bug`/`Move` non erano copy-safe** — sono value object immutabili e le
  `Position` sono singleton internati confrontati per identità: una copia non risultava
  uguale all'originale e ogni lookup falliva in silenzio. Era già sbagliato prima; è
  diventato visibile solo quando le Position hanno iniziato a referenziare i vicini.
- Un `"play "` di troppo in una mossa attesa di `test_mcts.py`.

---

## 3. `mcts_batch.py`: ripartito dal backup

Il backup **non era** una copia quasi identica: era la versione **sequenziale**
precedente. `mcts_batch.py` era una riscrittura con thread e virtual loss che non poteva
funzionare — `Board.copy()` inesistente, worker in Python puro sotto GIL con il
`tree_lock` tenuto per tutta la selezione, tre lock globali, e un `except:` nudo che
inghiottiva ogni errore del batch processor.

Il file ora è il design sequenziale del backup, con i suoi bug corretti e le idee buone
della versione parallela reinnestate:

| | |
|---|---|
| **tenuto** | riuso dell'albero fra mosse, ma con fallback a un albero nuovo invece del `raise` quando l'avversario gioca fuori dall'albero |
| **tenuto** | cache dei valori persistente fra le mosse (ora limitata) invece di svuotarla a ogni mossa |
| **scartato** | la virtual loss, che ha senso solo con discesa parallela |
| **corretto** | la cache viene consultata **prima** di costruire il grafo, non dopo: un hit pagava comunque `_data_from_board` sulla foglia e su ogni mossa legale |
| **corretto** | il mapping dei risultati era quadratico (test di appartenenza su lista dentro una comprehension) |
| **corretto** | il flush finale perdeva tutto ciò che restava sotto la soglia di batch |
| **corretto** | ripristinato il corto-circuito sui nodi già espansi, il cui lavoro di rete `expand()` buttava via |

`Board.copy()` è stato aggiunto (2,3 µs ora che le tabelle sono condivise) e condivide con
la copia le cache, che sono indicizzate su valori globali.

`bench/test_mcts_batch.py` verifica: mossa legale o `pass`, board restituita esattamente
com'era, riuso dell'albero su 6 ply, matti in uno che la gestione dei terminali vede, e
una ricerca ripetuta che con cache calda non ricostruisce **nessun** grafo.

---

## 4. Ottimizzazioni applicate

Ordinate per guadagno misurato. Ogni riga è stata verificata con `bench/diffcheck.py`.

| intervento | effetto misurato |
|---|---|
| Tabelle Zobrist condivise e piatte | `Board()` 7745× |
| Vicini precalcolati su `Position` + uso in DFS, conteggio regina, piazzamenti, cavallette | MCTS **1,95×** |
| `Bug.index` precalcolato (via `BugName[str(bug)]` e `Bug.__hash__` che costruiva stringhe) + niente `stringify_move` durante la ricerca | MCTS **1,46×** |
| Punti di articolazione: cache sulla **forma** dell'alveare invece che sullo Zobrist, indici densi, DFS iterativa | MCTS **1,73×** |
| Mosse scorrevoli e conteggio regina su indici interi | `get_valid_moves` a freddo **1,64×**, MCTS 1,08× |
| `_data_from_board`: costanti fuori dal loop, via il `pos_bug_to_index` mai letto, archi in numpy, niente `pin_memory` per-tensore | costruzione grafo **1,89×** (torch escluso) |
| Assert e logging fuori dai loop di rollout; `Random` non stringifica più tutte le mosse | incluso sopra |
| Tabelle precalcolate per `Direction`, mosse di scarabeo/pillbug su indici | **neutro** (1,00×): tenuto perché corretto, non perché più veloce |

Dettagli sul perché della cache "per forma": i punti di articolazione dipendono solo da
**quali celle sono occupate**. L'altezza delle pile, quale insetto sta dove e il turno
cambiano la chiave Zobrist senza cambiare la risposta, quindi la cache mancava di
continuo. La board mantiene ora un set delle celle occupate e un hash incrementale di
quel set, aggiornati in `safe_play`/`undo`. Dopo questa modifica Tarjan **sparisce dal
profilo**: era il 53%.

Altre correzioni lato torch, **non misurabili qui perché torch non è installato su questa
macchina**: flag TF32 spostati in `__init__` invece che risettati a ogni batch,
`model.eval()` solo quando serve invece che a ogni `predict`, `pin_memory` tolto dal
DataLoader CPU-only.

---

## 5. Cosa resta aperto

### Deciso e non fatto, con motivo

- **Riuso dell'albero anche in `brains.MCTS`.** `run_simulation_from` ricrea la radice a
  ogni mossa, buttando l'albero precedente; `MCTS_BATCH` il riuso ce l'ha. Non l'ho
  aggiunto perché cambia il comportamento della ricerca euristica usata da
  `gen_dataset`, e `MCTS_BATCH` è il searcher che conta per MCTS+GAT. Vale tipicamente
  20-40% di rollout utili.
- **Parallelismo.** Deciso: sequenziale. Il threading su Python puro non può dare
  speedup. La strada è **multi-processo** con la rete dietro un server di inferenza;
  il prerequisito era una `Board` leggera e serializzabile, che ora esiste.
- ~~**Colonna di feature inutilizzata**~~ — **non era un bug.** Avevo segnalato la
  colonna 1 sempre a zero come off-by-one. Ispezionando il dataset reale (vedi
  `TRAINING_and_DATASET.md`) risulta che il one-hot è a 9 slot con lo slot 0 riservato a
  "nessun insetto", e il dataset usa esattamente la stessa codifica: i due layout
  combaciano. Va lasciata così.

### Non ancora affrontato

- **`src/ai/training.py` non esiste** ma è importato da `gen_dataset/data_generator.py`,
  `match_generator.py`, `match_gen_parallel.py`, `graph_db_converter.py`. Quei quattro
  script non partono, ed è proprio la pipeline che serve per rigenerare il dataset.
  **È il prossimo blocco da sciogliere**, visto che il training va rifatto.
- **`torch.compile(mode="reduce-overhead")`** in `oracleGNN.__init__`: il `try/except`
  intorno non intercetta nulla (la compilazione è lazy, avviene alla prima forward), e i
  CUDA graph di `reduce-overhead` vogliono shape stabili, che le batch di MCTS non hanno.
  Da misurare contro `mode="default", dynamic=True` e contro nessuna compilazione.
- **Tabella di trasposizione sui nodi**: due cammini alla stessa posizione creano ancora
  due sottoalberi indipendenti.
- **Progressive widening**: a ogni espansione si valuta la rete su *tutte* le mosse
  legali (25-100 in mediogioco).
- **Potatura delle cache per-board**: `_snapshots`, `_snapshots_art_pos` e `_draw_counter`
  crescono senza limite; `_pos_to_bug` conserva le liste vuote delle celle abbandonate.
- `src/test/` contiene l'harness di duello, non test: nome fuorviante e conflitto con
  `pytest`. Nessun `pyproject.toml`: gli import dipendono dalla cwd.

---

## 6. Come verificare

```bash
python bench/bench_engine.py                   # numeri correnti
python bench/ab.py bc7e37a HEAD --rounds 6     # confronto col baseline
python bench/diffcheck.py bc7e37a .            # il motore non è cambiato
python bench/test_mcts_batch.py                # MCTS_BATCH end-to-end, senza torch
cd src && python -m compileall -q engine ai test gen_dataset
```

Tutto gira senza torch. Quando ci sarà una macchina con CUDA, restano da verificare lì:
il ramo `autocast`, il loop di training con l'ottimizzatore persistente, e `torch.compile`.
