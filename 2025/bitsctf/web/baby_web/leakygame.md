# LEAKY GAME

We were given a picture and a big challenge text. The words of the challenge text:

> "I just moved into this place, and my neighbor is... weird. A mad scientist—wild hair, crazy eyes, always muttering about 'the perfect game for my plan.' The other night, I saw a strange chessboard through his window, pieces arranged in a way that felt... wrong. The board was conquered by white and the game was weirdly verrry loong. But I think he's hiding something. Something big. Think of this and give me some clues."

As a *Phineas and Ferb* fan, I remembered something about Dr. Heinz Doofenshmirtz, and he looked exactly the same. Now I looked at the chess game—it wasn't a normal game. It had **9 pawns, 3 knights, and 3 same-colored bishops**.

## Step 1: Identifying the Chess Variant

This must be a chess variant! I concluded that this was **Crazyhouse**. The chessboard seemed to be from **Lichess**.

So, I used the **advanced search option in Lichess**, but I didn't know:
- The exact date the game was played.
- The time control used.

However, the challenge text mentioned that the game was *long*.

For the date, I took the creation date of the given image: **5 February**. To avoid risks, I set the search date from **4–6 February** and time control from **1 hour to 3 hours**. I analyzed every game but didn't find anything.

Then I realized that a game can be "big" in **number of turns** as well. So I analyzed around 300 games and finally found a game played by an account:

[DocGoof52025126](https://lichess.org/@/DocGoof52025126)

In his last game, at **move 53**, he was in the same position as in the given PNG. Also, his username resembled **Dr. Doof**.

## Step 2: Investigating the User

I analyzed all his **17 games** but found nothing. However, I noticed that he had **16 imported games**. Viewing the PGN of the top game, I found this:

```
[Event "TOP SECRET STUFF"]
[Site "https://lichess.org/p6TIg4qY"]
[Date "2025.02.04"]
[White "MasterBot1992"]
[Black "DrFeinzGoof4545"]
[Result "1-0"]
```

### Key Findings:
1. The **PGN files were edited**, and all **16 imported games** had different event names but no flag.
2. I now had **two related accounts**: `DocGoof52025126` and `DrFeinzGoof4545`.

## Step 3: Searching for the Accounts

I searched for these accounts on **every platform** possible. Finally, I found a **GitHub user** with the same ID:

[DrFeinzGoof4545](https://github.com/DrFeinzGoof4545)

He had a repository containing a **video** explaining how to store data in chess games:

[Deepest Secret Inator](https://github.com/DrFeinzGoof4545/deepest-secret-inator)

## Step 4: Extracting Hidden Data

I downloaded all necessary files and modified `decode.py` to extract the hidden data:

```python
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
```

I entered **every PGN manually** and got these outputs:

```
BITSCTF{H0p3_y0u_h4d_A_gr8_J0urn3y_eheheheheheh!??_564A8E9D}
```

## Conclusion

The **final flag** was:

```
BITSCTF{H0p3_y0u_h4d_A_gr8_J0urn3y_eheheheheheh!??_564A8E9D}
```

