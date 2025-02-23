import base64

f = open("rot3.txt", "r")
content = f.read()
f.close()
for i in range(40):
    content = base64.b64decode(content).decode()
    
    
print(content)