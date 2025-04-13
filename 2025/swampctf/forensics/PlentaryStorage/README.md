## Plentary Storage (Forensics)

### **Challenge:**

We were provided with a **LevelDB database** and needed to extract meaningful data.

### **Solution:**

- Used a **Python script** to iterate over the database and decode the stored values.
- Some values appeared to be **Base64-encoded**.
- Decoded the value twice to reveal the flag.

### **Python Script Used:**

```python
import plyvel
import json

db_path = "/mnt/c/Users/Tanish Kumar/Desktop/challenge/"
db = plyvel.DB(db_path, create_if_missing=False)

for key, value in db:
    print(f"Key: {key.decode(errors='ignore')}")
    
    try:
        decoded_value = value.decode(errors='ignore')
        json_data = json.loads(decoded_value)
        print(json.dumps(json_data, indent=4))  
    except (UnicodeDecodeError, json.JSONDecodeError):
        print(f"Raw Value: {value}")  

db.close()
```

### **Decoded Output:**

```
payload: "eyJrZXkiOiJcIjMzNTc5M2Q1LTRhYzEtNDgyMy05MmM3LWZkM2I1YTZhMmEwN1wiIiwib3AiOiJQVVQiLCJ2YWx1ZSI6ImV5SmtZWFJoSWpwYkluTjNZVzF3UTFSR2V6RndaalV0WWpRMU0yUXRaRFEzTkdJME5UTjlJbDBzSW1sa0lqb2lYQ0l6TXpVM09UTmtOUzAwWVdNeExUUTRNak10T1RKak55MW1aRE5pTldFMllUSmhNRGRjSWlKOSJ9"
```

- Decoding this **twice** with Base64 gave us the flag.

### **Flag:**

```
swampCTF{1pf5-b453d-d474b453}
```
