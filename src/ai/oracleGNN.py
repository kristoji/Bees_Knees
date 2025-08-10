from engine.board import Board
from engine.game import Move
from typing import Dict
import numpy as np
import torch
from torch_geometric.data import Data
from ai.graph_honored_network import GraphClassifier
from ai.improved_honored_GNN import GraphClassifierImproved
from ai.training import Training
from ai.oracle import Oracle
from ai.loader import GraphDataset
from gpt import board_to_simple_honored_graph
from engineer import Engine
from ai.log_utils import log_header, log_subheader
import os
from ai.brains import MCTS
from engine.enums import GameState, PlayerColor
from gpt import save_simple_honored_graph
import json

SHORT = 50
LONG = 100
SUPERLONG = 150
TURN_LIMIT = 100

class OracleGNN(Oracle):
    """
    Oracle that uses a neural network to predict the value and policy of a board state.
    """
    def __init__(self):
        #self.network = GraphClassifier(in_dim=13, hidden_dim=64, num_classes=1)

        self.device = torch.device(os.environ.get("TORCH_DEVICE", "cpu"))
        self.kwargs_network = {
            'conv_type': 'GAT'  # 'GIN', 'GAT', 'GCN'
        }
        self.network = GraphClassifierImproved(in_dim=13, hidden_dim=64, num_classes=1, **self.kwargs_network)
        self.network.to(self.device)
        self.cache: Dict[int, float] = {}
        
    def training(self, train_data_path:str, epochs:int) -> None:
        """
        Train the neural network with the provided training data.
        T is a tuple of (in_mats, out_mats, values)
        """
        self.path = train_data_path
        self.train_loader =  GraphDataset(folder_path=self.path) # ------------> DA METTERE co dataloader

        if not self.network:
            raise ValueError("Neural network is not initialized.")
        
        if self.device.type == 'cuda':
            print(f"Pre-loading dataset to GPU: {self.device}")
            self.train_loader = self._preload_to_gpu(self.train_loader)
        
        batch_size = 128
        self.train_loader = self.train_loader.get_dataloader(batch_size=batch_size, shuffle=True, num_workers=0)

        if not self.network:
            raise ValueError("Neural network is not initialized.")
        self.network.train_network(
            train_loader=self.train_loader,
            epochs=epochs,
        )

    def _preload_to_gpu(self, dataset):
        """Pre-move dataset to GPU to avoid repeated transfers."""
        for i in range(len(dataset)):
            if dataset.data[i] is not None:
                dataset.data[i] = dataset.data[i].to(self.device)
        return dataset
    
    def save(self, path: str) -> None:
        """
        Save weights
        """
        self.path = path
        self.network.save(path)

    def load(self, path: str) -> None:
        """
        Load weights
        """
        self.path = path
        self.network.load(path)

    def copy(self) -> 'OracleGNN':
        """
        Create a copy of the Oracle instance.
        """
        if not self.path:
            # self.save("temp.pth") # save in a temp file just to perform the copy
            raise ValueError("Path is not set. Cannot copy without a path.")
        new_oracle = OracleGNN()
        new_oracle.network.load(self.path) 
        return new_oracle
    
    def compute_heuristic(self, board: Board, game: bool = True) -> float:
        if board.state != GameState.IN_PROGRESS:
            if board.state == GameState.DRAW:
                v = 0.5
            else:
                v = 1.0 if (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK) else 0.0
        else:
            # v, _ = self.predict(board)    

            x, edge_index, pos_bug_to_index = board_to_simple_honored_graph(board)

            x = self.flatten_nodes(x)
            # Create PyTorch Geometric Data object
            data = Data(
                x = torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(),
                batch=torch.zeros(len(x), dtype=torch.long)  # Fixed: should be zeros for single graph        
                )
            data = data.to(self.device)  # Move to the correct device
            v = self.network.predict(data, use_sigmoid=True) # PREDICT THE STATE  
            v = self._to_float(v)
            # assert type(v) is float, f"Expected v to be float, got {type(v)}"
            # [INVERTED NET]
            v = 1 - v  
                  

        return v
    
    def flatten_nodes(self, x):
        temp = []
        for el in x:
            flattened = []
            for sub_el in el:
                if isinstance(sub_el, list):
                    flattened.extend(sub_el)  # More efficient than iterating
                else:
                    flattened.append(sub_el)
            temp.append(flattened)
        return temp

    def _to_float(self, x):

        if isinstance(x, torch.Tensor):
            # detach -> cpu -> squeeze -> item
            if x.numel() == 1:
                return float(x.detach().cpu().item())
            return float(x.detach().cpu().view(-1)[0].item())
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.squeeze())
            return float(x.reshape(-1)[0])
        return float(x)

    def predict(self, board: Board, game: bool = True) -> tuple[float, Dict[Move, float]]:
        """
        Predict the value and policy for the given board state.
        """

        # TODO : convert the board into a graph representation
        x, edge_index, pos_bug_to_index = board_to_simple_honored_graph(board)

        x = self.flatten_nodes(x)

        # Skip empty graphs -.--> TODO: we can do better?????
        if len(x) == 0:
            v = 0.5
        else:
            # Create PyTorch Geometric Data object
            data = Data(
                x = torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(),
                batch=torch.zeros(len(x), dtype=torch.long)  # Fixed: should be zeros for single graph        
                )
            data = data.to(self.device)  # Move to the correct device
            
            if not self.cache.get(board.zobrist_key, None):
                v = self.network.predict(data, use_sigmoid=True) # PREDICT THE STATE
                v = self._to_float(v)
                self.cache[board.zobrist_key] = v
            else:
                v = self.cache[board.zobrist_key]

        assert type(v) is float, f"Expected v to be float, got {type(v)}"
        # [INVERTED NET]
        v = 1-v

        valid_moves = list(board.get_valid_moves())
        pi = {}

        for m in valid_moves:
            
            board.safe_play(m) # safe to optimize

            if board.state != GameState.IN_PROGRESS:
                if board.state == GameState.DRAW:
                    V = 0.5
                else:
                    V = 1.0 if (board.state == GameState.WHITE_WINS and board.current_player_color == PlayerColor.WHITE) or (board.state == GameState.BLACK_WINS and board.current_player_color == PlayerColor.BLACK) else 0.0


                # V = torch.tensor([V], dtype=torch.float32, device=self.device)  # Ensure V is a tensor with Size 1
            else:
                x, edge_index, pos_bug_to_index = board_to_simple_honored_graph(board)
                x = self.flatten_nodes(x)
                data = Data(
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long).contiguous(),
                    batch=torch.zeros(len(x), dtype=torch.long)  # Fixed: should be zeros for single graph            
                    )
                data = data.to(self.device)  # Move to the correct device

                if not self.cache.get(board.zobrist_key, None):
                    V = self.network.predict(data, use_sigmoid=True)    # PREDICT THE STATE
                    V = self._to_float(V)
                    self.cache[board.zobrist_key] = V

                else:
                    V = self.cache[board.zobrist_key]

                V = 1 - V # [INVERTED NET]

            assert type(V) is float, f"Expected V to be float, got {type(V)}"
            pi[m] = 1 - V

            board.undo()
            
        # Softmax the probabilities
        if pi:

            probs = np.array(list(pi.values()))
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)
            pi = {move: prob for move, prob in zip(pi.keys(), probs)}
        else:
            pi={}
            #raise ValueError("No valid moves found in the board state.")

        if game and isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
            # Convert all pi tensor values to numpy
            pi = {move: prob.cpu().numpy() if isinstance(prob, torch.Tensor) else prob for move, prob in pi.items()} # TODO: prob()..() [0] ! prendere solo il val, non il ndarr

        return v, pi
    
    def generate_matches(self, iteration_folder: str, n_games: int = 500, n_rollouts: int = 1500, verbose: bool = False, perc_allowed_draws: float = float('inf')) -> None:

        engine = Engine()
        os.makedirs(iteration_folder, exist_ok=True)

        game = 1
        draw = 0
        wins = 0
        discarded = 0

        while game < n_games:

            log_header(f"Game {game} of {n_games}: {draw}/{wins} [D/W] - Discarded: {discarded}", width=70)

            T_game = []
            v_values = []

            engine.newgame(["Base+MLP"])
            s = engine.board
            mcts_game = MCTS(oracle=self, num_rollouts=n_rollouts)
            winner = None

            game_folder = os.makedirs(iteration_folder+f"/game_{game}/", exist_ok=True)

            turn = 1
            value = 1.0

            while not winner and num_moves <= TURN_LIMIT:

                mcts_game.run_simulation_from(s, debug=False)

                save_simple_honored_graph(move_idx=turn, board=s, save_dir=game_folder)

                v_values.append(value)
                value *= -1.0

                a: str = mcts_game.action_selection(training=True)

                engine.play(a, verbose=verbose)

                winner: GameState = engine.board.state != GameState.IN_PROGRESS

                num_moves += 1

            if engine.board.state == GameState.DRAW:
                draw += 1
                if draw > 2 * (perc_allowed_draws * n_games):
                    break
                if draw > perc_allowed_draws * n_games:
                    continue
            else:
                wins += 1

            log_subheader(f"Game {game} finished with state {engine.board.state.name}")

            final_value: float = 1.0 if engine.board.state == GameState.WHITE_WINS else -1.0 if engine.board.state == GameState.BLACK_WINS else 0.0

            v_values = [v * final_value for v in v_values]

            # for each json in game_dir, add the value
            for json_file in os.listdir(game_folder):
                if json_file.endswith('.json'):
                    json_path = os.path.join(game_folder, json_file)
                    with open(json_path, 'r') as f:
                        graph_data = json.load(f)
            #         # graph_data['v'] = v_values[int(json_file.split('_')[2].split('.')[0])]
                    #print(int(json_file.split('_')[1].split('.')[0])-1)
                    graph_data['v'] = v_values[int(json_file.split('_')[1].split('.')[0])-1]
                    with open(json_path, 'w') as f:
                        json.dump(graph_data, f)

            game += 1