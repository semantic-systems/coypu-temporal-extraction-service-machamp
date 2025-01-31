from subprocess import call
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
    input_filename = "input_file_D"+str(datetime.datetime.now()).replace(' ','T')+".bio"
    output_filename = "output_file_D"+str(datetime.datetime.now()).replace(' ','T')+".log"

    with open(input_filename, "w") as f:
        for token in tokens:
            f.write(token+'\tO\n')
        f.write("\n")

    call("python predict.py " + "finetuned_models/xlm-roberta_large/xlm-roberta-large_tempeval_multi/model.pt "
         + input_filename +" "+output_filename, shell=True)

    os.remove(input_filename)
    os.remove(output_filename+".eval")

    objects = []
    with open(output_filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objects.append({'token': line.split()[0], 'value': line.split()[1]})
    os.remove(output_filename)
    
    result = json.dumps({'results': objects})
    print(result)
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5989)
