#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import defaultdict, Counter
import json
from typing import List
from universal_ie.task_format.task_format import TaskFormat
from universal_ie.utils import tokens_to_str
from universal_ie.ie_format import Entity, Relation, Event, Label, Sentence, Span
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer


class TEMPEVALRELATION(TaskFormat):
    def __init__(self, sent_id, sentence_tokens, sentence_text, sentence_entities, sentence_relations, language="en"):
        super().__init__(language=language)
        self.sent_id = sent_id
        self.tokens = sentence_tokens
        self.entities = sentence_entities
        self.events = list()
        self.text = sentence_text
        self.relations = sentence_relations

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

        relations = []
        for span_index, relation in enumerate(self.relations):
            source = relation["args"][0]
            target = relation["args"][1]
            #Find corresponding entity
            source_entity = None
            target_entity = None
            for entity in entities:
                if entity.span.text == source["text"] and entity.label.label_name == source["type"]:
                    source_entity = entity
                if entity.span.text == target["text"] and entity.label.label_name == target["type"]:
                    target_entity = entity
            if source_entity is None or target_entity is None:
                print("Error: Relation source or target not found in entities")
            label = Label(relation["type"])
            uie_relation = Relation(
                arg1=source_entity,
                arg2=target_entity,
                label=label
            )
            relations += [uie_relation]

        return Sentence(
            tokens=self.tokens,
            entities=entities,
            relations=relations,
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
                relations = doc["relation"] if "relation" in doc else []

                sentence_id = counter["sentence_id"]
                counter["sentence_id"] += 1

                instance = TEMPEVALRELATION(
                    sent_id=sentence_id, 
                    sentence_tokens=tokens, 
                    sentence_text=text, 
                    sentence_entities=entities,
                    sentence_relations=relations,
                    language=language
                ).generate_instance()

                sentence_list += [instance]

        print(filename, counter)
        return sentence_list
