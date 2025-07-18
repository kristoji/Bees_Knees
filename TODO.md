# SETUP
Download di [Mzinga](https://github.com/jonthysell/Mzinga/tree/main)

Si interagisce con l'[Engine]((https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#engine-commands)) da terminale:
```
chmod +x Mzinga*
./MzingaEngine
# per vedere le info e le estensioni:
info 
# per i comandi:
help 
# per iniziare una nuova partita con le estensioni:
newgame Base+MLP 
# per vedere le mosse disponibili:
validmoves
# per fare una mossa (sono tutte mosse relative):
play mossa
# per la mossa migliore entro 5 sec:
bestmove time 00:00:05
```

Il MzingaViewer, invece, è un'interfaccia grafica che utilizza l'engine.

# Generate elf/exe
```
pyinstaller ./src/engine.py --name BeesKneesEngine --noconsole --onefile
```

# TODO
- test tra le engines rispetto ai cambi
- aggiungi hash.py per implementare zobrist hash come in Mzinga
- in board bisogna implementare l`hash ogni volta che cambia il turno, l`ultima mossa e la board
- implementare la transposition table per l`ab pruning
- in ai, la board ha già la lista di mosse. non serve salvare la mossa nel nodo perchè è già in board. allo stesso modo, non serve la lista di mosse: la board ha già la cache dove salva le mosse valide


# TODO.md
adj above, below
float su embedding


- Sistemare embedding caselle sopra
- Embedding nodi e archi in modo da avere un float
- risultato è la eval function della mossa passata in input.

dataset
(grafo con mossa, eval function)

Ordering assumption: By mapping bug types to a linearly scaled float (type_index/(N–1)), you introduce an arbitrary ordering among types. The network may learn unintended correlations (e.g. “type 3 is closer to type 4 than to type 0”). If you ever see weird clustering of those embeddings, you could switch to a learned embedding lookup (e.g. nn.Embedding(num_types, d)) so the network chooses the distances itself.