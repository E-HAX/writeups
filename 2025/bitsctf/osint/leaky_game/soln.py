from time import time
from math import log2
from chess import pgn, Board
from util import get_pgn_games

def decode(pgn_string: str, output_file_path: str):
    start_time = time()
    total_move_count = 0
    games = get_pgn_games(pgn_string)
    
    with open(output_file_path, "wb") as output_file:
        output_data = ""
        
        for game_index, game in enumerate(games):
            chess_board = Board()
            game_moves = list(game.mainline_moves())
            total_move_count += len(game_moves)
            
            for move_index, move in enumerate(game_moves):
                legal_move_ucis = [m.uci() for m in list(chess_board.generate_legal_moves())]
                move_binary = bin(legal_move_ucis.index(move.uci()))[2:]
                
                if game_index == len(games) - 1 and move_index == len(game_moves) - 1:
                    max_binary_length = min(int(log2(len(legal_move_ucis))), 8 - (len(output_data) % 8))
                else:
                    max_binary_length = int(log2(len(legal_move_ucis)))
                
                required_padding = max(0, max_binary_length - len(move_binary))
                move_binary = ("0" * required_padding) + move_binary
                chess_board.push_uci(move.uci())
                output_data += move_binary
                
                if len(output_data) % 8 == 0:
                    output_file.write(bytes([int(output_data[i * 8: i * 8 + 8], 2) for i in range(len(output_data) // 8)]))
                    output_data = ""
    
    print(f"Successfully decoded PGN with {len(games)} game(s), {total_move_count} total move(s) ({round(time() - start_time, 3)} sec)")

if __name__ == "__main__":
    pgn_string = """ENTER PGN HERE"""
    decode(pgn_string, "output.txt")
