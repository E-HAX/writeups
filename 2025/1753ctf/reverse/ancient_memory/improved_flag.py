import torch
import torch.nn as nn
import string

# Flag pattern and current best guess
KNOWN_PATTERN = "1753c{xxxxxxx_xx_xx_xxxxx}"
CURRENT_GUESS = "1753c{wrwtt3n_1n_my_bra1n}"
FLAG_LENGTH = len(KNOWN_PATTERN)

# Recreate the neural network model
class FlagSimilarityModel(nn.Module):
    def __init__(self):
        super(FlagSimilarityModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(FLAG_LENGTH * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Set device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Convert a flag to the tensor format expected by the model
def flag_to_tensor(flag, device):
    X = torch.zeros(FLAG_LENGTH, 128)
    for i, char in enumerate(flag):
        X[i, ord(char)] = 1
    return X.flatten().to(device)

# Load the model
def load_model(model_path):
    device = get_device()
    print(f"Using device: {device}")
    
    model = FlagSimilarityModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

# Test the similarity score of a flag
def test_flag(model, device, flag):
    with torch.no_grad():
        tensor = flag_to_tensor(flag, device)
        score = model(tensor).item()
    return score

# Test every position to find potential improvements
def refine_flag(model, device, current_flag):
    chars_to_try = string.ascii_lowercase + string.ascii_uppercase + string.digits + "_-{}!"
    best_flag = current_flag
    best_score = test_flag(model, device, current_flag)
    print(f"Starting refinement with flag: {current_flag}")
    print(f"Initial score: {best_score:.6f}")
    
    # Try different characters at each position
    improved = True
    iteration = 1
    
    while improved:
        improved = False
        print(f"\n--- Iteration {iteration} ---")
        
        for pos in range(6, len(current_flag)-1):  # Skip the prefix "1753c{" and suffix "}"
            current_char = current_flag[pos]
            
            for char in chars_to_try:
                if char == current_char:
                    continue
                    
                test_flag_str = current_flag[:pos] + char + current_flag[pos+1:]
                score = test_flag(model, device, test_flag_str)
                
                if score > best_score:
                    best_score = score
                    best_flag = test_flag_str
                    improved = True
                    print(f"Improved! Position {pos}: '{current_char}' -> '{char}', Score: {score:.6f}")
                    print(f"New flag: {best_flag}")
        
        if improved:
            current_flag = best_flag
            iteration += 1
            
    print(f"\nFinal refined flag: {best_flag}")
    print(f"Final score: {best_score:.6f}")
    
    # Check if we might need to try common substitutions
    if best_score < 0.99:
        print("\nTrying common letter substitutions...")
        common_substitutions = {
            'w': 'W', '3': 'e', '1': 'i', 'i': '1', 'e': '3',
            'a': '4', '4': 'a', 'o': '0', '0': 'o', 's': '5', '5': 's',
            't': '7', '7': 't', 'l': '1', 'b': '8', '8': 'b',
            'wr': 'w', 'tt': 't', 'tt': 'th'
        }
        
        for pos in range(6, len(current_flag)-1):
            current_char = current_flag[pos]
            
            # Try individual character substitutions
            for orig, subst in common_substitutions.items():
                if current_char == orig:
                    test_flag_str = current_flag[:pos] + subst + current_flag[pos+1:]
                    score = test_flag(model, device, test_flag_str)
                    
                    if score > best_score:
                        best_score = score
                        best_flag = test_flag_str
                        print(f"Substitution improved! Position {pos}: '{orig}' -> '{subst}', Score: {score:.6f}")
                        print(f"New flag: {best_flag}")
            
            # Try two-character substitutions
            if pos < len(current_flag) - 2:
                two_chars = current_flag[pos:pos+2]
                for orig, subst in common_substitutions.items():
                    if len(orig) == 2 and two_chars == orig:
                        test_flag_str = current_flag[:pos] + subst + current_flag[pos+2:]
                        score = test_flag(model, device, test_flag_str)
                        
                        if score > best_score:
                            best_score = score
                            best_flag = test_flag_str
                            print(f"Two-char substitution improved! Position {pos}: '{orig}' -> '{subst}', Score: {score:.6f}")
                            print(f"New flag: {best_flag}")
    
        print(f"\nAfter substitutions - Final flag: {best_flag}")
        print(f"Final score: {best_score:.6f}")

def manual_test(model, device):
    """Function to test specific flag variations"""
    test_cases = [
        "1753c{wrwtt3n_1n_my_bra1n}",  # Current best
        "1753c{wr1tt3n_1n_my_bra1n}",  # Changed 'w' to '1'
        "1753c{writt3n_1n_my_bra1n}",  # Changed 'wrwtt' to 'writt'
        "1753c{written_1n_my_bra1n}",  # No '3'
        "1753c{written_in_my_brain}",  # No leetspeak
        "1753c{wr1tten_1n_my_bra1n}",  # Partial leetspeak
        "1753c{writt3n_in_my_brain}",  # Partial leetspeak
    ]
    
    print("\nTesting specific variations:")
    for flag in test_cases:
        score = test_flag(model, device, flag)
        print(f"{flag}: {score:.6f}")
        
    # Allow custom input
    print("\nEnter custom flags to test (empty line to quit):")
    while True:
        custom_flag = input("> ")
        if not custom_flag:
            break
        if len(custom_flag) != FLAG_LENGTH:
            print(f"Warning: Flag length should be {FLAG_LENGTH} characters")
            continue
        score = test_flag(model, device, custom_flag)
        print(f"{custom_flag}: {score:.6f}")

if __name__ == "__main__":
    model_path = 'ancient-memory/model.pt'
    model, device = load_model(model_path)
    
    # First, test the current guess
    current_score = test_flag(model, device, CURRENT_GUESS)
    print(f"Current guess '{CURRENT_GUESS}' has score: {current_score:.6f}")
    
    # Try to refine the flag
    refine_flag(model, device, CURRENT_GUESS)
    
    # Allow manual testing of variations
    manual_test(model, device)