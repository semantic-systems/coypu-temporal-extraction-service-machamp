#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import defaultdict, Counter
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Event, Label, Sentence, Span
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer


class TEMPEVAL(TaskFormat):
    def __init__(self, sent_id, sentence_tokens, sentence_text, sentence_entities, language="en"):
        super().__init__(language=language)
        self.sent_id = sent_id
        self.tokens = sentence_tokens
        self.entities = sentence_entities
        self.events = list()
        self.text = sentence_text

    def generate_instance(self):
        entities = []
        for span_index, entity in enumerate(self.entities):
            idx = [i for i in range(entity["start"], entity["end"] + 1)]
            ent = Entity(
                    span=Span(
                        tokens=[word_tokenize(t) for t in entity["text"]],
                        indexes=idx,
                        text=entity["text"],
                        text_id=self.sent_id
                    ),
                    label=Label(entity['type']),
                    text_id=self.sent_id,
                    record_id=str(self.sent_id) + "#%s" % span_index if self.sent_id else None) #     0#1
            entities += [ent]

        return Sentence(
            tokens=self.tokens,
            entities=entities,
            events=list(),
            text_id=self.sent_id,
        )

    @staticmethod
    def load_from_file(filename, language="en") -> List[Sentence]:
        sentence_list = []
        counter = Counter()

        with open(filename) as fin:
            for line in fin:
                doc = json.loads(line.strip())
                entities = doc["entity"]
                tokens = doc["tokens"]
                text = doc["text"]

                sentence_id = counter["sentence_id"]
                counter["sentence_id"] += 1

                instance = TEMPEVAL(sent_id=sentence_id, sentence_tokens=tokens, sentence_text=text, sentence_entities=entities,language=language
                ).generate_instance()

                sentence_list += [instance]

        print(filename, counter)
        return sentence_list
    

class TEMPEVALENTITY(TaskFormat):
    #Only for the TempEval Relation Converter converted datasets of tempeval
    def __init__(self, sent_id, sentence_tokens, sentence_text, sentence_entities, language="en"):
        super().__init__(language=language)
        self.sent_id = sent_id
        self.tokens = sentence_tokens
        self.entities = sentence_entities
        self.events = list()
        self.text = sentence_text

    def generate_instance(self):
        entities = []
        for span_index, entity in enumerate(self.entities):
            idx = entity["offset"]
            ent = Entity(
                span=Span(
                    tokens=word_tokenize(entity["text"]),
                    indexes=idx,
                    text=entity["text"],
                    text_id=self.sent_id
                ),
                label=Label(entity['type']),
                text_id=self.sent_id,
                record_id=str(self.sent_id) + "#%s" % span_index if self.sent_id else None
            ) 
            entities += [ent]


        return Sentence(
            tokens=self.tokens,
            entities=entities,
            relations=[],
            events=list(),
            text_id=self.sent_id,
        )

    @staticmethod
    def load_from_file(filename, language="en") -> List[Sentence]:
        sentence_list = []
        counter = Counter()

        with open(filename) as fin:
            for line in fin:
                doc = json.loads(line.strip())
                entities = doc["entity"]
                tokens = doc["tokens"]
                text = doc["text"]

                sentence_id = counter["sentence_id"]
                counter["sentence_id"] += 1

                instance = TEMPEVALENTITY(
                    sent_id=sentence_id, 
                    sentence_tokens=tokens, 
                    sentence_text=text, 
                    sentence_entities=entities,
                    language=language
                ).generate_instance()

                sentence_list += [instance]

        print(filename, counter)
        return sentence_list
