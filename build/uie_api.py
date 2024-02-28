from subprocess import PIPE, run
from flask import Flask, request
import nltk
import datetime
import os
import json
import logging

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_temporal_extraction():
    sentence = request.json.get('sentence')
    command = ["python", "inference_sentence.py", "--sentence", sentence]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if not result.stdout:
        app.logger.error('There was an error when analyzing the sentence. The following output was given:'+result.stderr)
    result = json.dumps({'results': result.stdout})
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5929)
