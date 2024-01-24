from subprocess import PIPE, run
from flask import Flask, request
import nltk
import datetime
import os
import json

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_temporal_extraction():
    sentence = request.json.get('sentence')
    tokens = nltk.word_tokenize(sentence)
    
    command = ["python", "inference_sentence.py", "--sentence", sentence]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    result = json.dumps({'results': result.stdout})
    print(result)
    return result


#get_temporal_extraction('Today I will go.')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5989)
