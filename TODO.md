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
