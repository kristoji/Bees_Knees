# In questa versione usiamo una lista di coppie (move, probability) invece di un dict per ogni mossa
# così da ridurre l'overhead di molti oggetti dict annidati.

import os
import re
import json
from datetime import datetime
from ai.training import Training
from engine.enums import GameState
from engineer import Engine

PRO_MATCHES_FOLDER = "pro_matches/"
GAME_TO_PARSE = 1000


def parse_hive_game(file_path: str) -> list[str]:
    moves = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['):
                continue
            parts = line.split('.')
            if len(parts) > 1:
                san = parts[1].strip()
                moves.append(san)
    return moves


def generate_matches_list_pv(source_folder: str):
    engine = Engine()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"data/{ts}/pro_matches"
    os.makedirs(save_dir, exist_ok=True)

    for game_idx, fname in enumerate(os.listdir(source_folder), start=1):
        engine.newgame(["Base+MLP"])
        board = engine.board
        pv_list = []  # lista di (pi_list, value) per ogni mossa
        value = 1.0

        moves = parse_hive_game(os.path.join(source_folder, fname))
        for san in moves:
            valid = board.get_valid_moves()
            parsed = board._parse_move(san)
            # creiamo una lista di tuple (mossa_str, prob)
            pi_list = [(san, 1.0 if mv == parsed else 0.0) for mv in valid]
            pv_list.append((pi_list, value))
            value *= -1.0
            engine.play(san)

        # risultato finale e moltiplicatore
        outcome = engine.board.state
        final_mult = (1.0 if outcome == GameState.WHITE_WINS else
                      -1.0 if outcome == GameState.BLACK_WINS else
                      0.0)
        pv_list = [ (plist, v*final_mult) for plist, v in pv_list ]

        # salviamo su file JSON
        path = os.path.join(save_dir, f"game_{game_idx}_pv.json")
        with open(path, 'w') as f:
            json.dump(pv_list, f)

        if game_idx >= GAME_TO_PARSE:
            break


def load_pv_list(file_path: str):
    """
    Restituisce una lista di (pi_list, value),
    dove pi_list è [(move_str, prob), ...]
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
