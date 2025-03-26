# Solution
```
# Input the text as a single string
input_text = input().strip()
num_groups = int(input().strip())
shifts = list(map(int, input().strip().strip('[]').split(', ')))

def decrypt_text(text, shifts):
    decrypted = []
    index = 0
    alpha_chars = [c for c in text if c.isalpha()]
    
    for shift in shifts:
        group = alpha_chars[index:index+5]  # Extracting group of 5 or remaining chars
        decrypted_group = ''
        for char in group:
            new_char = chr(((ord(char) - ord('a') - shift + 26) % 26) + ord('a'))
            decrypted_group += new_char
        decrypted.append(decrypted_group)
        index += 5
    
    decrypted_text = list(text)
    alpha_index = 0
    
    for i in range(len(decrypted_text)):
        if decrypted_text[i].isalpha():
            decrypted_text[i] = decrypted[alpha_index // 5][alpha_index % 5]
            alpha_index += 1
    
    return ''.join(decrypted_text)

# Decrypt the input text
decoded_text = decrypt_text(input_text, shifts)
print(decoded_text)
```