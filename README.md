# Bees_Knees

## Description
[UHP](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol)-compliant [Hive](https://en.wikipedia.org/wiki/Hive_(game)) in Python, inspired by [CrystalSpider](https://github.com/Crystal-Spider/hivemind) and [jonthysell](https://github.com/jonthysell/Mzinga/tree/main).

## Usage
The engine executable can be created with the following command:
```
pyinstaller ./src/engine.py --name BeesKneesEngine --noconsole --onefile
```

Then, the engine can be used from the terminal or inside the [MzingaViewer](https://github.com/jonthysell/Mzinga/releases/tag/v0.15.1).

## Install the requirements
```
pip install -r requirements.txt
```

##  Use the Makefile
```
make data #generates the dataset
make k-means #generates clusters
make train #fine-tunes the LLM
make duel #plays Hive
```
