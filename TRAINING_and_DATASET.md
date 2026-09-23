# Dataset e training della GNN — analisi

Ispezione di `Hive_dataset.rar` (Drive, 314 MB) e revisione del setup di training.
**Nessun training è stato lanciato.** Del dataset ho scaricato solo **6 MB** (i primi 4 e
gli ultimi 2), che bastano per leggere gli header dell'archivio ed estrarre 259 partite
complete di campione.

---

## 0. Verdetto in breve

**Il dataset è utilizzabile e il formato è già quello giusto.** Su 259 partite estratte,
i grafi che il codice attuale genera dalla stessa posizione combaciano **esattamente**:
colore, one-hot del tipo, `pinned`, `pinning` e **tutti gli archi**. Le etichette `v`
sono corrette e nella convenzione giusta.

Ci sono tre difetti, tutti circoscritti e tutti sistemabili **senza riscaricare nulla e
senza riparare `gen_dataset/`**, perché ogni cartella partita contiene il `board.txt` con
la partita completa: il dataset si può rigenerare in locale dal motore attuale.

| # | problema | gravità |
|---|---|---|
| 1 | La feature "punto di articolazione" è **vecchia di un ply** | alta — è l'unica colonna sbagliata, e all'inferenza sarebbe corretta |
| 2 | `move_N.json` conta i ply **saltando i pass**, non è `ply N-1` | media — sbaglia l'allineamento sul 4% delle partite |
| 3 | `gen_dataset/graph_db_converter.py` oggi produrrebbe `v` **invertito** | alta — solo se rigeneri con quel codice |

Sul training invece il problema principale **non è un bug**: è che la rete è
*value-only* e il prior della policy viene schiacciato quasi a uniforme. Vedi §4.

---

## 1. Cosa c'è nell'archivio

`Hive_dataset.rar`, 314 MB, formato RAR5.

```
HIVE-DB_GRAPH-<N>_<collezione>/
    game_<K>/
        board.txt        gamestring UHP completa della partita
        move_1.json      posizione prima della 1ª mossa non-pass
        move_2.json      ...
    BAK/graph_cache_pikkola.pkl
```

Collezioni viste negli header: `GRAPH-1..4` × `{tournament, humans, bots}`, più un
`GRAPH-1_bots-after-2023`. Quindi **non è un corpus omogeneo**: ci sono partite di
tornei, di umani su Boardspace e di bot.

Stime (densità delle voci misurata su testa e coda dell'archivio, che variano parecchio):
**ordine di 10⁴ partite e ~0,5–1,1 M posizioni**. La numerazione `game_N` arriva almeno
a 2816 in una singola collezione.

Statistiche sul campione di 259 partite (`GRAPH-1_tournament`):

| | |
|---|---|
| esiti | WhiteWins 148, BlackWins 104, Draw 7 |
| lunghezza partita | min 20, media 56, max 328 ply |
| nodi per grafo | media 16,6, max 28 |
| posizioni con grafo vuoto | 1,8% (scartate dal loader, è la posizione iniziale) |
| bilanciamento etichette | `v=+1` 7102, `v=-1` 7067, `v=0` 425 |

`BAK/` è già ignorato da `GraphDataset` (`if "BAK" in subfolder: continue`) e il `.pkl`
che contiene non viene raccolto dal glob, che guarda solo il livello superiore.

---

## 2. Compatibilità col loader: cosa combacia

Il JSON ha esattamente le tre chiavi che `GraphDataset._process_data` si aspetta —
`x`, `edge_index`, `v` — e niente altro.

**Feature dei nodi.** `x` è una riga per nodo nella forma
`[colore, [one-hot a 9], pinned, pinning, art]`, che appiattita dà **13 feature**:
`in_dim=13` è corretto. Il one-hot ha 9 slot per 8 tipi di insetto perché lo slot 0 è
riservato a "nessun insetto" e non viene mai acceso.

> Nel report precedente avevo segnalato la colonna 1 sempre a zero come un off-by-one da
> correggere. **Mi ero sbagliato**: il dataset usa la stessa codifica a 9 slot, quindi i
> due layout combaciano. Va lasciata com'è, e ho corretto il commento nel codice.

**Etichetta `v`.** Alterna ±1 dentro la partita, ed è **relativa al giocatore di turno**:
`+1` se chi deve muovere poi vincerà, `-1` se perderà, `0` per patta. Verificato su tutte
le posizioni di due partite (48/48 e 49/49). `_process_data` fa `(v+1)/2` → `{0, 0.5, 1}`,
che è esattamente la convenzione usata all'inferenza (`compute_heuristic` restituisce
"probabilità che vinca chi muove"). **Nessun problema di prospettiva.**

**Verifica diretta.** Ho riprodotto il grafo dal `board.txt` col motore attuale e l'ho
confrontato con quello del dataset su 40 partite (~2400 posizioni):

```
archi diversi:           0
colonne 0..11 diverse:   0
colonna 12 diversa:      su ~50% delle posizioni
```

---

## 3. I tre difetti, e come sistemarli

### 3.1 La colonna "articulation" è vecchia di un ply

L'unica colonna che diverge è la 12. Non è rumore e non è invertita: a livello di
singolo nodo l'accordo col motore è **93,7%**, e la base rate coincide (50,1% nel
dataset, 50,8% nel motore). È **sfasata di una mossa**:

```
art del dataset == art del motore sulla stessa posizione :  35,1%
art del dataset == art del motore un ply PRIMA           :  63,4%   <--
nessuna delle due                                        :   1,5%
```

(Il 35% non è "corretto per caso": è semplicemente l'insieme dei casi in cui l'ultima
mossa non ha cambiato i punti di articolazione, e quindi le due risposte coincidono.)

Esempio concreto, `game_1/move_4.json` — tre pezzi: `wL(0,0)`, `bL(0,-1)`, `wQ(-1,1)`.
`bL` e `wQ` non sono adiacenti, quindi togliendo `wL` l'alveare si spezza: `wL` **è** un
punto di articolazione. Il dataset non ne segna nessuno — che è la risposta giusta per
la posizione *precedente*, quella a due pezzi.

**Non è colpa della mia riscrittura di Tarjan**: ho confrontato `_art_pos` fra il codice
pre-ottimizzazioni e quello attuale su tutta `game_1` e sono identici (0 differenze). Il
dataset è stato generato da una versione ancora precedente, in cui `_art_pos` veniva
letto prima di essere aggiornato.

**Perché conta.** È l'unica feature che dice "questo pezzo è bloccato dalla regola
dell'alveare unico" — informazione di valore alto e non deducibile localmente dal resto
del grafo. Ma soprattutto: **all'inferenza il motore la calcola correttamente**, quindi
allenare sulla versione sfasata crea uno scarto di distribuzione fra training e gioco su
una feature su tredici.

### 3.2 `move_N.json` salta i pass

Il generatore incrementa il contatore solo sulle mosse diverse da `pass`, ma gioca tutte
le mosse. Quindi `move_N` è la posizione **prima della N-esima mossa non-pass**, non
"dopo N-1 ply". Sul campione: 15 partite su 259 (5,8%) contengono almeno un `pass`, e in
10 di esse l'assunzione ingenua disallinea alcune posizioni.

Va tenuto presente in qualunque ricostruzione: si contano i ply saltando i `pass`.

### 3.3 Se rigeneri col codice attuale, `v` esce invertito

In `src/gen_dataset/graph_db_converter.py:generate_matches`:

```python
value = -1.0
for move_idx, san in enumerate(moves):
    if san != 'pass':
        ...
        v_values.append(value)      # la 1ª posizione salvata prende -1.0
    value *= -1.0
...
final_mult = 1.0 if outcome == GameState.WHITE_WINS else -1.0 if ... else 0.0
v_values = [v * final_mult for v in v_values]
```

La prima posizione salvata è quella col Bianco al tratto e riceve `-1.0`. Per una partita
`WhiteWins`, `final_mult = +1`, quindi resta `-1.0`: "chi muove perde", mentre il Bianco
ha vinto. Il dataset sul Drive ha `+1.0`, cioè è **giusto**; è il codice nel repo a essere
sbagliato (o comunque diverso da quello con cui il dataset è stato prodotto).

Una rete allenata su etichette invertite impara a predire che chi muove *perde*, e MCTS
sceglierebbe sistematicamente le mosse peggiori — con una win rate contro il random
*sotto* il 50%, che è il sintomo da cercare.

### 3.4 La correzione: rigenerare in locale dai `board.txt`

Ogni cartella partita contiene il `board.txt` con la partita completa, e ho verificato
che il motore attuale riproduce esattamente colore, tipo, pinned, pinning e archi.
Quindi **non serve né `gen_dataset/` né riscaricare**: basta rileggere i `board.txt`,
rigiocarli e riscrivere gli `x` con il `_data_from_board` attuale.

Lo script è breve. Per ogni `game_*/board.txt`:

1. parsare il gamestring: `type;state;turn;mossa1;mossa2;…`
2. ricostruire `Board("Base+MLP")` e rigiocare le mosse una per una
3. tenere un contatore `k` che avanza **solo sulle mosse diverse da `pass`**; prima di
   giocare la k-esima mossa non-pass, scrivere il grafo in `move_k.json`
4. `v` = `+1` se il giocatore di turno è quello che vince secondo `state`, `-1` se perde,
   `0` se `Draw` — oppure semplicemente riusare il `v` già presente nel JSON, che è
   corretto (è il modo più sicuro: evita di reintrodurre il bug del segno)

In più, visto che si sta rigiocando tutto, conviene salvare anche **la mossa
effettivamente giocata** come indice sulla lista delle mosse legali: è il target di
policy che oggi manca, e senza rigiocare non è recuperabile. Vedi §4.2.

Costo: ~10⁶ posizioni × ~50 µs di `_data_from_board` ≈ un'ora single-core, banalmente
parallelizzabile per partita.

---

## 4. Il training: ha senso?

### Cosa è corretto

- Il target è il risultato finale dal punto di vista di chi muove, in `{0, 0.5, 1}`, con
  `BCEWithLogitsLoss` su un output scalare. È il classico value head di AlphaZero e va
  bene con i target morbidi delle patte.
- Le convenzioni di segno sono coerenti lungo tutta la catena: dataset → `(v+1)/2` →
  sigmoid → `compute_heuristic` → `score = 1 - V` in MCTS → `1 - node.reward()` in
  backprop. Le ho ripercorse una per una, non c'è inversione.
- Le classi sono bilanciate (50,1% / 49,9%).
- La rappresentazione a grafo è sensata: un nodo per pezzo (non per cella), archi fra
  vicini alla stessa altezza più archi verticali per le pile.

### 4.1 Il problema statistico: 10⁴ etichette indipendenti, non 10⁶

Questo è il punto più importante dopo il prior.

Tutte le ~56 posizioni di una partita condividono **lo stesso bit di informazione** (chi
ha vinto), con il segno che alterna. Non sono 10⁶ esempi indipendenti: sono **~10⁴
etichette indipendenti**, ciascuna replicata ~56 volte su posizioni quasi identiche fra
loro (differiscono di una mossa).

La rete configurata in `train_gnn.py` (GIN, 6 layer, `hidden_dim=256`, `mlp_layers=3`) ha
**~1,25 M parametri**. 1,25 M parametri contro ~10⁴ etichette indipendenti, per **75
epoche**: l'overfitting non è un rischio, è lo scenario di default.

E **non lo vedresti**, perché lo split è sbagliato:

```python
train_dataset, test_dataset = random_split(self, [train_size, test_size])
```

`random_split` divide le **posizioni**, non le partite. Posizioni della stessa partita
finiscono sia in train che in validation, e condividono l'etichetta. La validation loss
è quindi ottimisticamente distorta e non misura generalizzazione. In più `random_split`
è chiamato senza `generator`, quindi **lo split cambia a ogni run**: due training non
sono confrontabili e riprendere un training fa leak.

**Da fare, in ordine:**
1. **split per partita**, non per posizione, con seed fisso;
2. tenere anche un test set di partite mai viste, separato dalla validation;
3. ridurre drasticamente le epoche e tenere il checkpoint con la validation migliore
   (oggi si salva ogni 5 epoche e alla fine, senza guardare la validation);
4. valutare `hidden_dim` più piccolo (64–128) e weight decay più alto di `1e-6`.

### 4.2 Il problema per MCTS: il prior è quasi uniforme

Questo è **il motivo principale per cui MCTS sembrerebbe debole anche con una value net
buona**, ed è un problema di design, non un bug.

La rete non ha una policy head. MCTS costruisce il prior valutando **tutti i figli** un
ply avanti e facendo la softmax dei valori:

```python
scores.append(1 - V)      # V = sigmoid(logit) ∈ (0,1)
arr = np.exp(arr - np.max(arr)); arr /= np.sum(arr)
```

Con la sigmoid, `scores` sta in `[0,1]`. Una softmax su un intervallo largo 1 dà un
rapporto **massimo** fra la mossa migliore e la peggiore di `e¹ ≈ 2,72`. In pratica i
valori di mosse sorelle staranno entro 0,1–0,2 l'uno dall'altro, quindi il rapporto reale
è ~1,1–1,2. Con 40 mosse legali il prior uniforme è 0,025 e il migliore arriva forse a
0,028: **il prior non porta quasi informazione**, e l'albero esplora quasi a caso finché
le statistiche `Q` non prendono il sopravvento.

C'è già un commento nel codice che mostra che l'idea era nota:

```python
# graph_network.py, predict()
# rimuoviamo sigmoide così che i valori unbounded vanno diretti alla softmax.
results = logits if not use_sigmoid else torch.sigmoid(logits)
```

…ma MCTS passa `use_sigmoid=True` ovunque, quindi la sigmoide non viene mai rimossa.

**Due rimedi, dal più economico al più efficace:**

- **Subito, gratis:** usare i **logit** (non la sigmoid) per la softmax della policy, con
  una temperatura `T` calibrata, e tenere la sigmoid solo per il valore della foglia.
  Una riga in `_collect_leaf_for_batch` più una `T` da tarare.
- **La vera soluzione:** aggiungere una **policy head**. Il target ce l'hai: nelle partite
  pro la mossa giocata è la mossa "giusta", ed è recuperabile rigiocando i `board.txt`
  (§3.4). Due benefici:
  - il prior diventa informativo;
  - **l'espansione passa da `b+1` valutazioni di rete a 1**. Con `b ≈ 25–60` in mediogioco
    è un fattore 25–60× sul costo per espansione, cioè molto più di tutto quello che ho
    ottimizzato nel motore.

  Il generatore originale calcolava già `pi` come one-hot sulla mossa giocata
  (`graph_db_converter.py:41`) e c'era un `save_graph` con `move_adj`, ma è commentato:
  la versione usata è `save_simple_honored_graph`, che la policy non la salva.

### 4.3 Altri punti minori

- **`pooling='add'`** su grafi con 13–28 nodi: l'embedding scala col numero di pezzi. Con
  un target di valore può andare, ma rende la rete sensibile al conteggio dei pezzi in
  modo non voluto. Vale la pena provare `mean` o `concat` come ablazione.
- **`use_residual=False`** in `train_gnn.py` con 6 layer GIN: senza residui e con sole
  LayerNorm, 6 layer di message passing su grafi di diametro ~8 sono al limite
  dell'over-smoothing. Vale un'ablazione a 3–4 layer.
- L'accuratezza calcolata in `forward()` confronta `logits > 0` con un target morbido:
  per le patte (`y=0.5`) non è mai "giusta". È comunque scartata.
- `GraphDataset` carica **tutti** i grafi in una lista Python in RAM e poi li serializza
  in un unico `.pkl`. Su ~10⁶ grafi sono diversi GB e un pickle enorme; e
  `_preload_to_gpu` li sposta **tutti** sulla GPU, cosa che a quel volume va quasi
  sicuramente in OOM. Per il cluster serve un dataset su disco (uno shard per collezione,
  o `InMemoryDataset` di PyG con `.pt` pre-collati).
- **Mescolare `tournament`, `humans` e `bots`** in un'unica cartella è una scelta che
  `GraphDataset` fa automaticamente (itera tutte le sottocartelle di `folder_path`). Le
  partite di bot deboli insegnano valutazioni sbagliate. Suggerisco di partire da
  `tournament` + `humans` di livello alto, e tenere `bots` come esperimento separato.

---

## 5. Cosa farei, in ordine

1. **Scaricare ed estrarre l'archivio** su una macchina con spazio (314 MB compressi,
   verosimilmente diversi GB espansi — sono ~10⁶ file piccoli, occhio al filesystem).
2. **Rigenerare gli `x`** dai `board.txt` con il motore attuale (§3.4), riusando i `v`
   già presenti e salvando in più la mossa giocata come target di policy.
3. **Non toccare `gen_dataset/`** per ora: serve solo se vuoi ripartire dai PGN. Se ci
   torni, prima va risolto l'import di `ai.training` e il segno di `v` (§3.3).
4. **Split per partita** con seed fisso, e un test set di partite mai viste.
5. **Prima run onesta**: rete piccola (`hidden_dim=64–128`, 3–4 layer), poche epoche,
   early stopping sulla validation per-partita. Serve a stabilire che la rete impara
   *qualcosa* — la baseline da battere è l'euristica `Oracle` già nel repo, che conta i
   vicini della regina.
6. **Poi** la policy head, che è dove sta il guadagno grosso per MCTS.
7. Misura finale che conta davvero: **win rate di MCTS+GNN contro MCTS+`Oracle` euristico
   a parità di rollout**. Loss e accuratezza di validation dicono poco su quanto bene
   giochi l'albero.

---

## 6. Cosa non ho fatto

- Non ho lanciato alcun training, e questa macchina non ha GPU né torch installato.
- Non ho scaricato il dataset completo: 6 MB su 314, da cui 259 partite di campione. Le
  conclusioni su formato, etichette e feature sono verificate su quel campione (~2400
  posizioni confrontate una per una); le stime di **dimensione totale** sono
  estrapolazioni dalla densità degli header e possono sbagliare di un fattore 2.
- Non ho ispezionato le collezioni `humans` e `bots`: il campione viene tutto da
  `GRAPH-1_tournament`. Presumo lo stesso formato, visto che li produce lo stesso
  generatore, ma **non l'ho verificato**.
- Non ho scritto lo script di rigenerazione: §3.4 ne descrive i passi, dimmi se lo vuoi.
