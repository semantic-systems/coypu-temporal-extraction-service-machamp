from subprocess import call, PIPE, run
from flask import Flask, request
import nltk
import datetime
import os
import json
import logging

app = Flask(__name__)

@app.route('/machamp', methods=['POST'])
def get_machamp_temporal_extraction():
    """
        It obtains the sentence sent in the body and does the following:
            - Tokenizes the sentence
            - Create an input file and an output file with the timestamp and the necessary type for the model
            - In the input file write the tokens of the sentence
            - The input and output files are put in the calling of the prediction of the MACHAMP model
            - Both the input and the generated .EVAL output files are deleted
            - Finally, the result is read from the output file and returned
    Returns
    -------
        JSON Object of the temporal extraction result from the MACHAMP model
    """
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

    return {'results': objects}

@app.route('/uie', methods=['POST'])
def get_uie_temporal_extraction():
    """
        With the sentence obtained from the body of the call, is inserted in the calling to the inference script of the
        UIE model. The output of this call is turned into a JSON and returned
    Returns
    -------
        JSON Object of the temporal extraction result from the UIE model
    """
    sentence = request.json.get('sentence')
    command = ["python", "inference_sentence.py", "--sentence", sentence]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if not result.stdout:
        app.logger.error('There was an error when analyzing the sentence. The following output was given:'+result.stderr)
    app.logger.error(result.stdout)
    app.logger.error(str(result.stdout))
    return {'results': json.loads(result.stdout)}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
