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