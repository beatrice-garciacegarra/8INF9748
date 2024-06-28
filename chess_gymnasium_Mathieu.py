import gym
import gym_chess
import chess
import random
import numpy as np
from collections import defaultdict
import unicodedata
from gym import spaces
import copy

class ChessEnvironment(gym.Env):
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        # King doesn't have a value because it can't be captured
    }

    def __init__(self):
        super(ChessEnvironment, self).__init__()
        self.board = chess.Board()
        self.previous_piece_counts = self.count_pieces()
        self.action_space = spaces.Discrete(4672) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.int32)
        self.last_move = None

    def count_pieces(self):
        piece_counts = {
            piece: {
                'white': len(self.board.pieces(piece, chess.WHITE)),
                'black': len(self.board.pieces(piece, chess.BLACK))
            } for piece in self.PIECE_VALUES
        }
        return piece_counts

    def is_terminal(self):
        return self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_insufficient_material()

    def reset(self):
        self.board.reset()
        self.previous_piece_counts = self.count_pieces()
        self.last_move = None
        return self.get_observation()

    def get_observation(self):
        observation = np.zeros((8, 8, 12), dtype=np.int32)
        for piece_type in range(1, 7):
            for color in [chess.WHITE, chess.BLACK]:
                bitboard = self.board.pieces(piece_type, color)
                for square in bitboard:
                    row, col = divmod(square, 8)
                    plane = (piece_type - 1) + (6 if color == chess.BLACK else 0)
                    observation[row][col][plane] = 1
        return observation

    def render(self):
        unicode_board = self.board.unicode(invert_color=True, borders=False)
        
        if self.last_move:
            board_array = unicode_board.split('\n')
            
            from_square = self.last_move.from_square
            to_square = self.last_move.to_square
            
            from_rank = 7 - (from_square // 8)
            from_file = from_square % 8
            to_rank = 7 - (to_square // 8)
            to_file = to_square % 8

            for i in range(len(board_array)):
                board_array[i] = board_array[i].replace('⭘', '.')
        
            from_square_char = board_array[from_rank][from_file*2]
            from_square_char_width = unicodedata.east_asian_width(from_square_char)

            if from_square_char_width in ('F', 'W'):
                from_square_char_end = from_file*2 + 2
            else:
                from_square_char_end = from_file*2 + 1

            to_square_char = board_array[to_rank][to_file*2]
            to_square_char_width = unicodedata.east_asian_width(to_square_char)

            if to_square_char_width in ('F', 'W'):
                to_square_char_end = to_file*2 + 2
            else:
                to_square_char_end = to_file*2 + 1

            if self.board.turn == chess.WHITE:
                background_color_from = "\033[47m"
                background_color_to = "\033[42m"
            else:
                background_color_from = "\033[47m" 
                background_color_to = "\033[41m"

            board_array[from_rank] = (
                board_array[from_rank][:from_file*2] +
                background_color_from + from_square_char + '\033[0m' +
                board_array[from_rank][from_square_char_end:]
            )

            # Only highlight the from square if it's not on the same rank as the to square
            if from_rank != to_rank:
                board_array[to_rank] = (
                    board_array[to_rank][:to_file*2] + 
                    background_color_to + to_square_char + '\033[0m' + 
                    board_array[to_rank][to_square_char_end:]
                )

            unicode_board = '\n'.join(board_array)

        print(unicode_board, '\n\nScore', 'WHITE :' if self.board.turn == chess.BLACK else 'BLACK :', self.get_reward(), '\n')

    def legal_moves(self):
        return list(self.board.legal_moves)

    def decode(self, action):
        legal_moves = list(self.board.legal_moves)
        if isinstance(action, int) and action < len(legal_moves):
            return legal_moves[action]
        else:
            raise ValueError("Invalid action")

    def get_reward(self):
        current_piece_counts = self.count_pieces()
        reward = 0

        for piece, value in self.PIECE_VALUES.items():
            captured_white = self.previous_piece_counts[piece]['white'] - current_piece_counts[piece]['white']
            captured_black = self.previous_piece_counts[piece]['black'] - current_piece_counts[piece]['black']

            if self.board.turn == chess.BLACK:
                reward += (captured_black - captured_white) * value
            else:
                reward += (captured_white - captured_black) * value

        if self.board.is_checkmate():
            reward += 100 if self.board.turn == chess.BLACK else -100
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0

        return reward
    
    def move(self, action):
        move = action if isinstance(action, chess.Move) else chess.Move.from_uci(action)
        self.board.push(move)
        self.last_move = move

class MonteCarloTreeSearchNode:

    def __init__(self, env, parent=None, parent_action=None):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._rewards = 0
        self._number_of_visits = 0
        self._untried_moves = self.untried_moves()

    def untried_moves(self):
        return self.env.legal_moves()

    def q(self):
        return self._rewards / self._number_of_visits if self._number_of_visits else 0

    def n(self):
        return self._number_of_visits

    def expand(self):
        if not self._untried_moves:
            raise ValueError("No more untried moves to expand.")
        
        action = self._untried_moves.pop()
        next_env = copy.deepcopy(self.env)
        next_env.move(action)
        child_node = MonteCarloTreeSearchNode(next_env, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.env.is_terminal()

    def rollout(self):
        current_rollout_env = copy.deepcopy(self.env)
        while not current_rollout_env.is_terminal():
            action = self.rollout_policy(current_rollout_env.legal_moves())
            current_rollout_env.move(action)
        return current_rollout_env.get_reward()

    def backpropagate(self, reward):
        self._number_of_visits += 1
        self._rewards += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def is_fully_expanded(self):
        return len(self._untried_moves) == 0

    def best_child(self, c_param=0.1):
        epsilon = 1e-6
        choices_weights = [
            (c.q()) + c_param * np.sqrt((2 * np.log(self.n() + epsilon)) / (c.n() + epsilon)) for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, simulation_no=45):
        for _ in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.0)  # c_param=0.0 to choose the most visited child
    
    def move(self, action):
        move = chess.Move.from_uci(action.uci())
        env_copy = copy.deepcopy(self.env)
        if move in env_copy.board.legal_moves:
            env_copy.board.push(move)
        else:
            raise ValueError("Invalid move: " + str(move))
        return env_copy
    
    def update_env(self, env):
        self.env = env
        self._untried_moves = self.untried_moves()

def main():
    env = ChessEnvironment()
    env.reset()

    root_white = MonteCarloTreeSearchNode(env=env)
    root_black = MonteCarloTreeSearchNode(env=env)

    epochs = 10

    for _ in range(epochs):
        while not env.is_terminal():
            best_child = None

            if env.board.turn == chess.WHITE:
                best_child = root_white.best_action()
                root_white = best_child
                
            else:
                best_child = root_black.best_action()
                root_black = best_child

            env = best_child.env         

            env.render()

            if env.is_terminal():
                break
            else:
                root_white.update_env(env)
                root_black.update_env(env)

        if env.board.is_checkmate():
            winner = "White" if env.board.turn == chess.BLACK else "Black"
            print(f"Checkmate! {winner} wins.")
        elif env.board.is_stalemate():
            print("Stalemate! The game is a draw.")
        elif env.board.is_insufficient_material():
            print("Insufficient material! The game is a draw.")
        elif env.board.is_seventyfive_moves():
            print("Seventy-five moves rule! The game is a draw.")
        elif env.board.is_fivefold_repetition():
            print("Fivefold repetition! The game is a draw.")

        env.reset()
        root_white.update_env(env)
        root_black.update_env(env)

if __name__ == "__main__":
    main()
