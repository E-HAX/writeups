# MMDLX

given a MMDLX.txt file

it has some text, that looks like base64 but not actually base64

i tried running ceaser cipher on it. (using rot13.com)

I uploaded the content of given text file on [rot13.com](https://rot13.com/)

Saved the data with various shifts (1..26) in different text files.

One of the file (with shift of 3) when decoded using base64 40 times gives us the flag.

(40 times base64 decoding hint we got from the fact that. 40*64 = 2560 which is MMDLX in roman numeric sytem)

Flag - `KashiCTF{w31rd_numb3r5_4nd_c1ph3r5}`


Script -

```
import base64

f = open("rot3.txt", "r")
content = f.read()
f.close()
for i in range(40):
    content = base64.b64decode(content).decode()
  
  
print(content)
```
