from inference import HuggingfacePredictor, post_processing
import argparse
import transformers as huggingface_transformers
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
import nltk
from uie.sel2record.record import MapConfig
from uie.sel2record.sel2record import SEL2Record
import json

TIMEX_BEGIN = "<TIMEX3>"
TIMEX_END = "</TIMEX3>"

def tokenize_sentences(sentences, sent_tokenizer):
    """
    Tokenizes given sentences into individual tokens using a provided sentence tokenizer.
    
    Args:
        sentences (str): The input sentence to be tokenized.
        sent_tokenizer: The pre-loaded sentence tokenizer object.
    
    Returns:
        list: A list of tokens extracted from the sentence.
    """
    tokens = []
    
    #tokens = [token for sent in sent_tokenizer.tokenize(sentence) for token in word_tokenize(sent)]
    for sent in sent_tokenizer.tokenize(sentences):
        print(sent)
        for token in word_tokenize(sent):
            tokens.append(token)
    return tokens

if __name__ == "__main__":
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./finetuned_models/base/tempeval_multi')
    parser.add_argument('--max_source_length', default=256, type=int)
    parser.add_argument('--max_target_length', default=192, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('-c', '--config', dest='map_config', help='Offset Re-mapping Config', default='config/offset_map/closest_offset_en.yaml')
    parser.add_argument('--decoding', default='spotasoc')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--match_mode', default='normal', choices=['set', 'normal', 'multimatch'])
    options = parser.parse_args()

    model_path = options.model
    predictor = HuggingfacePredictor(
        model_path=model_path,
        schema_file=f"./etc/temporal_schema/record.schema",
        max_source_length=options.max_source_length,
        max_target_length=options.max_target_length,
    )

    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    tokenizer = huggingface_transformers.T5TokenizerFast.from_pretrained(
         model_path)
    
    map_config = MapConfig.load_from_yaml(options.map_config)
    schema_dict = SEL2Record.load_schema_dict("./etc/temporal_schema")
    sel2record = SEL2Record(
        schema_dict=schema_dict,
        decoding_schema=options.decoding,
        map_config=map_config,
    )
    
    while(True):
        # Eample input: "Yesterday there was a large thunderstorm in Hannover, Germany from 4pm to 10pm. It was the largest storm recorded in Germany since July 1996."
        input_sentence = input("Input sentence:\n> ")
        input_tokens = tokenizer(input_sentence)
        print("Analyzing input...")

        output_seq2seq = predictor.predict([input_sentence])
        output_seq2seq = post_processing(output_seq2seq[0])

        tokens = tokenize_sentences(input_sentence, sent_tokenizer)
        record = sel2record.sel2record(pred=output_seq2seq, text=input_sentence, tokens=tokens)
        tagged_tokens = tokens.copy()

        for type, index in record["entity"]["offset"]:
            start_index = index[0]
            end_index = index[1] if len(index) >= 2 else index[0]
            tagged_tokens[start_index] = f"{TIMEX_BEGIN}{tagged_tokens[start_index]}"
            tagged_tokens[end_index] = f"{tagged_tokens[end_index]}{TIMEX_END}"
        
        tagged_sentence = ""
        for token in tagged_tokens:
            if token not in [",", ".", "?", "!"]:
                tagged_sentence += f" {token}"
            else:
                tagged_sentence += token
        tagged_sentence = tagged_sentence.strip()

        print("\nResults:\n%s\n\n\n"
            % json.dumps(
                {
                    "input_text": input_sentence,
                    "tokens": tokens,
                    "seq2seq": output_seq2seq,
                    "record": record,
                    "tagged_sentence": tagged_sentence
                },
                indent = 4
            )
        )