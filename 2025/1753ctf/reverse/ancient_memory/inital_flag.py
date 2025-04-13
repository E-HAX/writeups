import torch
import torch.nn as nn
import itertools
import string

# Flag format from the original code
FLAG_PREFIX = "1753c{"
FLAG_SUFFIX = "}"
KNOWN_PATTERN = "1753c{xxxxxxx_xx_xx_xxxxx}"
FLAG_LENGTH = len(KNOWN_PATTERN)

# Recreate the neural network model (must match the original)
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

# Test a character at a specific position
def test_char_at_position(model, device, current_flag, position, chars_to_try):
    best_score = 0
    best_char = None
    
    for char in chars_to_try:
        test_flag = current_flag[:position] + char + current_flag[position+1:]
        
        with torch.no_grad():
            tensor = flag_to_tensor(test_flag, device)
            score = model(tensor).item()
            
            if score > best_score:
                best_score = score
                best_char = char
    
    return best_char, best_score

def recover_flag(model_path):
    model, device = load_model(model_path)
    
    # Start with the known pattern
    current_flag = KNOWN_PATTERN
    
    # First, identify the positions we need to recover
    positions_to_recover = [i for i, char in enumerate(current_flag) if char == 'x']
    
    # Characters to try (alphanumeric plus common symbols)
    chars_to_try = string.ascii_lowercase + string.ascii_uppercase + string.digits + "_-{}!"
    
    print(f"Starting flag recovery with pattern: {current_flag}")
    print(f"Need to recover {len(positions_to_recover)} positions")

    # Iteratively recover each position
    for pos in positions_to_recover:
        best_char, best_score = test_char_at_position(model, device, current_flag, pos, chars_to_try)
        
        # Update the current flag with the best character
        current_flag = current_flag[:pos] + best_char + current_flag[pos+1:]
        print(f"Position {pos}: Best char '{best_char}' with score {best_score:.6f}")
        print(f"Current flag: {current_flag}")

    # Verify the final flag
    with torch.no_grad():
        tensor = flag_to_tensor(current_flag, device)
        final_score = model(tensor).item()
    
    print(f"\nRecovered flag: {current_flag}")
    print(f"Final similarity score: {final_score:.6f}")
    
    # If the score is high enough, we likely found the correct flag
    if final_score > 0.99:
        print("High confidence in the recovered flag!")
    else:
        print("Warning: Low confidence in the recovered flag. Might need further refinement.")

if __name__ == "__main__":
    model_path = 'ancient-memory/model.pt'  # Adjust if your model file has a different path
    recover_flag(model_path)