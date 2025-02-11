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
