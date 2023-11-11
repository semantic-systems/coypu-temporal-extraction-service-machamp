import re
import copy
from collections import Counter
from typing import List, Match, Any, Tuple
import nltk
from .preprocessing_utils import DatasetNltkTokenizer, initiate_tokenizers
import os

class TagMakeinstance:
    #<MAKEINSTANCE eventID="e26" eiid="ei395" tense="NONE" aspect="NONE" polarity="POS" pos="NOUN"/>
    def __init__(self, event_id, eiid, tense, aspect, polarity, pos, modality) -> None:
        self.event_id = event_id
        self.eiid = eiid
        self.tense = tense
        self.aspect = aspect
        self.polarity = polarity
        self.pos = pos
        self.modality = modality


class TagTlink:
    #<TLINK lid="l1" relType="BEFORE" eventInstanceID="ei1" relatedToTime="t1" relatedToEventInstance="ei2"/>
    def __init__(self, lid, relType, eventInstanceID, timeID, relatedToTime, relatedToEventInstance, modality, signalID) -> None:
        self.lid = lid
        self.relType = relType
        self.eventInstanceID = eventInstanceID
        self.timeID = timeID
        self.relatedToTime = relatedToTime
        self.relatedToEventInstance = relatedToEventInstance
        self.modality = modality
        self.signalID = signalID
        
        self.tlink_type = None
        if self.timeID and self.relatedToTime:
            self.tlink_type = "TT"
        elif self.eventInstanceID and self.relatedToEventInstance:
            self.tlink_type = "EE"
        elif self.eventInstanceID and self.relatedToTime:
            self.tlink_type = "ET"
        elif self.timeID and self.relatedToEventInstance:
            self.tlink_type = "TE"
        else:
            raise Exception("Unknown tlink type")    


class TagEvent:
    #EVENT eid="e39" class="I_STATE">believe</EVENT>
    def __init__(self, eid, event_class, text) -> None:
        self.eid = eid
        self.event_class = event_class
        self.text = text
        self.position_in_sentence_start = None
        self.position_in_sentence_end = None

    def set_position_in_sentence(self, start: int, end: int) -> None:
        """
        Position of the temporal tokens in the sentence
        """
        self.position_in_sentence_start = start
        self.position_in_sentence_end = end


class TagFullEvent:
    #Event tag enriched with makeinstance tag
    def __init__(self, eid, event_class, eiid, tense, aspect, polarity, pos, modality, text, event_object) -> None:
        self.eid = eid
        self.event_class = event_class
        self.text = text
        self.eiid = eiid
        self.tense = tense
        self.aspect = aspect
        self.polarity = polarity
        self.pos = pos
        self.event_object = event_object
        self.modality = modality
        self.position_in_sentence_start = None
        self.position_in_sentence_end = None

    def __init__(self, event_object, makeinstance_object) -> None:
        self.eid = makeinstance_object.event_id
        self.event_class = event_object.event_class
        self.text = event_object.text
        self.eiid = makeinstance_object.eiid
        self.tense = makeinstance_object.tense
        self.aspect = makeinstance_object.aspect
        self.polarity = makeinstance_object.polarity
        self.pos = makeinstance_object.pos
        self.event_object = event_object
        self.modality = makeinstance_object.modality
        self.position_in_sentence_start = None
        self.position_in_sentence_end = None

    def set_position_in_sentence(self, start: int, end: int) -> None:
        """
        Position of the temporal tokens in the sentence
        """
        self.position_in_sentence_start = start
        self.position_in_sentence_end = end


class TagTimex3:
    """
    Example: <TIMEX3 tid="t100" type="DURATION" value="P2Y" mod="APPROX" temporalFunction="false" functionInDocument="NONE">
    """
    def __init__(self, tid, timex3_type, value, mod, temporal_function, function_in_document, text) -> None:
        self.tid = tid
        self.timex3_type = timex3_type
        self.value = value
        self.mod = mod
        self.temporalFunction = temporal_function if temporal_function else ""
        self.functionInDocument = function_in_document if function_in_document else ""
        self.text = text
        self.is_dct = FUNCTION_DOCUMENT_CREATION_TIME in self.functionInDocument.upper()
        self.is_dpt = FUNCTION_DOCUMENT_PUBLICATION_TIME in self.functionInDocument.upper()
        self.position_in_sentence_start = None
        self.position_in_sentence_end = None

    def set_position_in_sentence(self, start: int, end: int) -> None:
        """
        Position of the temporal tokens in the sentence
        """
        self.position_in_sentence_start = start
        self.position_in_sentence_end = end


class TagSignal:
    """
    Example: <SIGNAL sid="s156">until</SIGNAL>
    """
    def __init__(self, sid, text) -> None:
        self.sid = sid
        self.text = text


class TagDocumentMetaTime:
    #<DOCNO> ABC<TIMEX3 tid="t82" type="DATE" value="1998-01-08" temporalFunction="false" functionInDocument="CREATION_TIME">19980108</TIMEX3>.1830.0711 </DOCNO>
    def __init__(self, dct_timex3_object, dpt_timex3_object) -> None:
        self.dct_object = dct_timex3_object
        self.dct_id = dct_timex3_object.tid if dct_timex3_object else None

        self.dpt_object = dpt_timex3_object
        self.dpt_id = dpt_timex3_object.tid if dpt_timex3_object else None


class Document:
    def __init__(self, filename, filepath, clean_text, sentences, makeinstance_tags, tlink_tags, event_tags, timex3_tags, intra_sentence_relations, inter_sentence_relations, dct_relations, dpt_relations, dct, dpt) -> None:
        self.filename = filename
        self.filepath = filepath
        self.clean_text = clean_text

        self.sentences = sentences

        self.makeinstance_tags = makeinstance_tags
        self.tlink_tags = tlink_tags
        self.event_tags = event_tags
        self.timex3_tags = timex3_tags

        self.intra_sentence_relations = intra_sentence_relations
        self.inter_sentence_relations = inter_sentence_relations
        self.dct_relations = dct_relations
        self.dpt_relations = dpt_relations

        self.document_creation_time = dct
        self.document_publication_time = dpt


class Sentence:
    """
    Example:
    {
        "type": "during",
        "args": 
            [
                {
                    "type": "event",
                    "offset": [
                        31
                    ],
                    "text": "said"
                },
            
                {
                    "type": "duration",
                    "offset": [
                        44,
                        45
                    ],
                    "text": "during july"
                }
            ]
    }
    """
    def __init__(self, text, tagged_text, sentence_in_document, tokens, timex3_in_sentence, events_in_sentence, entity_relations, dct_relations, dpt_relations, labeled_entities) -> None:
        self.text = text
        self.tagged_text = tagged_text
        self.sentence_in_document = sentence_in_document #Sentence number in document
        self.tokens = tokens
        self.timex3_in_sentence = timex3_in_sentence #With positions
        self.events_in_sentence = events_in_sentence #With positions
        self.entity_relations = entity_relations
        self.dct_relations = dct_relations
        self.dpt_relations = dpt_relations
        self.labeled_entities = labeled_entities #For convenience and debugging, not needed elsewhere



class TemporalRelation:
    def __init__(self, relation_type, source, target, tlink, relation_code) -> None:
        self.relation_type = relation_type
        self.source = source
        self.target = target
        self.tlink = tlink
        self.relation_code = relation_code


"""
Constants
"""
FUNCTION_DOCUMENT_CREATION_TIME: str = "CREATION_TIME"
FUNCTION_DOCUMENT_PUBLICATION_TIME: str = "PUBLICATION_TIME"


"""
Text cleaning
"""
TEXT_CLEANING_PATTERNS = [
    ("\s{2,}", " "),
    ("\t", " ")
]

TAGS_TO_DELETE = [
    "NUMEX",
    "ENAMEX",
    "TURN",
    "CARDINAL",
    "SIGNAL",
    "NG",
    "AP",
    "HEAD",
    "VG",
    "PG",
    "RG",
    "VG-VBG",
    "VG-INF",
    "VG-VBN",
    "JG",
    "COMMENT",
    "IN-MW"
]

WORD_TOKENIZER, SENTENCE_TOKENIZER = initiate_tokenizers()


"""
Regex patterns to extract the contents of the tml files
"""
REGEX_SPAN_TEXT = r"<TEXT>.+?</TEXT>"
REGEX_SPAN_TIMEX3 = r"<TIMEX3[^>]*?>.+?</TIMEX3>"
REGEX_SPAN_EVENT = r"<EVENT[^>]*?>.+?</EVENT>"
REGEX_SPAN_SIGNAL = r"<SIGNAL[^>]*?>.+?</SIGNAL>"
REGEX_SPAN_SENTENCE = r"<s>.+?</s>"
REGEX_SPAN_TIMEML = r"<TimeML[^>]*?>.*?</TimeML>"
REGEX_SINGLETON_MAKEINSTANCE = r"<MAKEINSTANCE[^>]+?>"
REGEX_SINGLETON_TLINK = r"<TLINK[^>]+?>"
REGEX_SINGLETON_XML = r"<\?xml[^>]+>"

REGEX_XML_START_TIMEX3 = r"<TIMEX3[^>]+?>"
REGEX_XML_END_TIMEX3 = r"</TIMEX3>"
REGEX_XML_START_EVENT = r"<EVENT[^>]+?>"
REGEX_XML_END_EVENT = r"</EVENT>"

REGEX_ANY_XMLTAG = r"<[^>]+>"


"""
Special tokens
"""
EVENT_SPECIAL_TOKEN_START = " STARTEVENTXML "
EVENT_SPECIAL_TOKEN_END = " ENDEVENTXML "
TIMEX3_SPECIAL_TOKEN_START = " STARTTIMEX3XML "
TIMEX3_SPECIAL_TOKEN_END = " ENDTIMEX3XML "



def fetch_object(object: Match[str]) -> str:
    """
    Fetches the object from the regex match object.

    Args:
        object (Match[str]): regex match object.

    Returns:
        str: Object or None if object wasn't found.
    """
    if object:
        return object.group().split("=")[1].replace('"', "").upper().strip()
    else:
        return None
    

def read_attribute_value(regex: str, text: str) -> str:
    text_match = re.search(regex, text)
    return fetch_object(text_match)
    

def extract_makeinstance_objects(input_text: str) -> List[TagMakeinstance]:
    full_makeinstance_tags = re.findall(REGEX_SINGLETON_MAKEINSTANCE, input_text, re.IGNORECASE)
    makeinstance_tags = []
    for makeinstance_tag in full_makeinstance_tags:
        makeinstance_tags += [_create_makeinstance_object(makeinstance_tag)]
    return makeinstance_tags


def _create_makeinstance_object(text_makeinstance_tag: str) -> TagMakeinstance:
    """
    Example: <MAKEINSTANCE eventID="e48" eiid="ei414" tense="PRESENT" aspect="NONE" polarity="POS" pos="VERB" modality="can"/>
    """
    event_id = read_attribute_value(r'eventID="[^"]+?"', text_makeinstance_tag)
    eiid = read_attribute_value(r'eiid="[^"]+?"', text_makeinstance_tag)
    tense = read_attribute_value(r'tense="[^"]+?"', text_makeinstance_tag)
    aspect = read_attribute_value(r'aspect="[^"]+?"', text_makeinstance_tag)
    polarity = read_attribute_value(r'polarity="[^"]+?"', text_makeinstance_tag)
    pos = read_attribute_value(r'pos="[^"]+?"', text_makeinstance_tag)
    modality = read_attribute_value(r'modality="[^"]+?"', text_makeinstance_tag)

    return TagMakeinstance(event_id, eiid, tense, aspect, polarity, pos, modality)


def extract_tlink_objects(input_text: str) -> List[TagTlink]:
    full_tlink_tags = re.findall(REGEX_SINGLETON_TLINK, input_text, re.IGNORECASE)
    tlink_tags = []
    for tlink_tag in full_tlink_tags:
        tlink_tags += [_create_tlink_object(tlink_tag)]
    return tlink_tags


def _create_tlink_object(text_tlink_tag: str) -> TagTlink:
    """
    Example (ET): <TLINK lid="l3" relType="IS_INCLUDED" eventInstanceID="ei380" relatedToTime="t85"/>
    Example (TE): <TLINK lid="l4" relType="INCLUDES" timeID="t85" relatedToEventInstance="ei381"/>
    Example (TT): <TLINK lid="l5" relType="BEFORE" timeID="t86" relatedToTime="t82"/>
    Example (EE): <TLINK lid="l10" relType="BEFORE" eventInstanceID="ei387" relatedToEventInstance="ei386" signalID="s13"/>
    """
    lid = read_attribute_value(r'lid="[^"]+?"', text_tlink_tag)
    rel_type = read_attribute_value(r'relType="[^"]+?"', text_tlink_tag)
    event_instance_id = read_attribute_value(r'eventInstanceID="[^"]+?"', text_tlink_tag)
    time_id = read_attribute_value(r'timeID="[^"]+?"', text_tlink_tag)
    related_to_time = read_attribute_value(r'relatedToTime="[^"]+?"', text_tlink_tag)
    related_to_event_instance = read_attribute_value(r'relatedToEventInstance="[^"]+?"', text_tlink_tag)
    modality = read_attribute_value(r'modality="[^"]+?"', text_tlink_tag)
    signal_id = read_attribute_value(r'signalID="[^"]+?"', text_tlink_tag)

    return TagTlink(lid, rel_type, event_instance_id, time_id, related_to_time, related_to_event_instance, modality, signal_id)


def extract_event_objects(input_text: str) -> List[TagTimex3]:
    full_event_tags = re.findall(REGEX_SPAN_EVENT, input_text, re.IGNORECASE)
    event_tags = []
    for event_tag in full_event_tags:
        event_tags += [_create_event_object(event_tag)]
    return event_tags


def _read_tag_content(tag: str) -> str:
    """
    Removes the xml tags from the tag.

    Args:
        tag (str): The tag to remove the xml tags from.

    Returns:
        str: The tag content without xml tags.
    """
    return re.sub(REGEX_ANY_XMLTAG, "", tag).strip()


def _create_event_object(text_event_tag: str) -> TagEvent:
    """
    Example: <EVENT eid="e33" class="OCCURRENCE">owns</EVENT>
    Example: <EVENT eid="e371" class="STATE">thirty</EVENT>
    Example: <EVENT eid="e46" class="OCCURRENCE">goes</EVENT>
    """
    eid = read_attribute_value(r'eid="[^"]+?"', text_event_tag)
    event_class = read_attribute_value(r'class="[^"]+?"', text_event_tag)
    text = _read_tag_content(text_event_tag)
    
    return TagEvent(eid, event_class, text)


def extract_timex3_objects(input_text: str) -> List[TagTimex3]:
    full_timex3_tags = re.findall(REGEX_SPAN_TIMEX3, input_text, re.IGNORECASE)
    timex3_tags = []
    for timex3_tag in full_timex3_tags:
        timex3_tags += [_create_timex3_object(timex3_tag)]
    return timex3_tags


def _create_timex3_object(text_timex3_tag: str) -> TagTimex3:
    """
    Example: <TIMEX3 tid="t196" type="DATE" value="PRESENT_REF" temporalFunction="true" functionInDocument="NONE" anchorTimeID="t82">Now</TIMEX3>
    Example: <EVENT eid="e371" class="STATE">thirty</EVENT>
    Example: <EVENT eid="e46" class="OCCURRENCE">goes</EVENT>
    """
    tid = read_attribute_value(r'tid="[^"]+?"', text_timex3_tag)
    timex3_type = read_attribute_value(r'type="[^"]+?"', text_timex3_tag)
    value = read_attribute_value(r'value="[^"]+?"', text_timex3_tag)
    mod = read_attribute_value(r'mod="[^"]+?"', text_timex3_tag)
    temporal_function = read_attribute_value(r'temporalFunction="[^"]+?"', text_timex3_tag)
    function_in_document = read_attribute_value(r'functionInDocument="[^"]+?"', text_timex3_tag)
    text = _read_tag_content(text_timex3_tag)
    
    return TagTimex3(tid, timex3_type, value, mod, temporal_function, function_in_document, text)


def find_matching_pairs(list_a: List[Any], list_b: List[Any], key_a: str, key_b: str) -> List[tuple]:
    pairs = []
    copy_list_b = copy.deepcopy(list_b)
    for entry_a in list_a:
        for entry_b in copy_list_b:
            if getattr(entry_a, key_a) == getattr(entry_b, key_b):
                pairs += [(entry_a, entry_b)]
                del copy_list_b[copy_list_b.index(entry_b)]
    return pairs


def extract_full_event_objects(input_text: str) -> List[TagFullEvent]:
    event_objects = extract_event_objects(input_text)
    makeinstance_objects = extract_makeinstance_objects(input_text)
    event_makeinstance_pairs = find_matching_pairs(event_objects, makeinstance_objects, "eid", "event_id")

    full_event_objects = []
    for event_object, makeinstance_object in event_makeinstance_pairs:
        full_event_objects += [TagFullEvent(event_object, makeinstance_object)]
   
    return full_event_objects


def extract_signal_objects(input_text: str) -> List[TagSignal]:
    full_signal_tags = re.findall(REGEX_SPAN_SIGNAL, input_text, re.IGNORECASE)
    signal_tags = []
    for signal_tag in full_signal_tags:
        signal_tags += [_create_signal_object(signal_tag)]
    return signal_tags


def _create_signal_object(text_signal_tag: str) -> TagSignal:
    sid = read_attribute_value(r'sid="[^"]+?"', text_signal_tag)
    text = _read_tag_content(text_signal_tag)
   
    return TagSignal(sid, text)


def extract_meta_document_time(document_timex3_labels: List[TagTimex3]) -> TagDocumentMetaTime:
    #Keep track of the ids, to make sure that there is maximum one DCT and one DPT
    dct_timex3_objects = []
    dpt_timex3_objects = []
    dct_timex3 = None
    dpt_timex3 = None
    for timex3_label in document_timex3_labels:
        if timex3_label.is_dct:
            if getattr(timex3_label, "tid") not in dct_timex3_objects:
                dct_timex3_objects += [timex3_label.tid]
                dct_timex3 = timex3_label
        elif timex3_label.is_dpt:
            if getattr(timex3_label, "tid") not in dpt_timex3_objects:
                dpt_timex3_objects += [timex3_label.tid]
                dpt_timex3 = timex3_label

    if len(dct_timex3_objects) > 1:
        raise Exception("Multiple distinct DCTs found in document")
    if len(dpt_timex3_objects) > 1:
        raise Exception("Multiple distinct DPTs found in document.")
    
    return TagDocumentMetaTime(dct_timex3, dpt_timex3)


def extract_article_content_timebank_timeml(input_text: str) -> str:
    """
    Article content of the TimeMl files aren't wrapped in <TEXT> tags.
    They have to be found by rules. Heuristics found, that the article
    content starts at the first line, that contains an EVENT tag.
    Further, it ends at one line before the first occurence of 
    MAKEINSTANCE (most of the time) or TLINK.
    In some cases there are shorten broken strings above the
    MAKEINSTANCE tags that should be excluded. This function
    walks the end pointer down from the last EVENT tag but
    doesn't go further than MAKEINSTANCE or TLINK.

    The DCT is usually above the article content.
    """
    lines = input_text.split("\n")

    start_index = -1
    for index, line in enumerate(lines):
        if "<EVENT" in line.upper():
            start_index = index
            break

    end_index = -1
    for index, line in enumerate(lines):
        if "<MAKEINSTANCE" in line.upper() or "<TLINK" in line.upper():
            end_index = index
            break

    #Find line that contains last EVENT tag
    for index in range(end_index, 0, -1):
        line = lines[index]
        if "<EVENT" in lines[index].upper():
            end_index = index
            break

    treshold = 4
    for index in range(end_index, len(lines)):
        line = lines[index]
        split_parts = len(line.split(" "))
        line_contains_scope = "(" in line or ")" in line

        if "<MAKEINSTANCE" in line.upper() or "<TLINK" in line.upper():
            break

        if split_parts <= treshold and line_contains_scope:
            end_index = index
            break

    if start_index != -1 and end_index != -1:
        article_content = "\n".join(lines[start_index:end_index])
        article_content = article_content.strip("\n \t") 

        article_content = article_content.replace("\n", " ")
        article_content = re.sub(r"\s{2,}", " ", article_content)
        return article_content
    else:
        raise Exception("Couldn't find article content")
    

def strip_article_content_timebank_timeml(input_text: str) -> str:
    tagless_text = re.sub(REGEX_ANY_XMLTAG, "", input_text, flags=re.IGNORECASE)
    cleaned_text = _clean_text(tagless_text).strip()
    return cleaned_text


def extract_article_content_timebank_extra(input_text: str) -> str:
    full_text = input_text.replace("\n", " ")
    full_text = re.sub(r"\s{2,}", " ", full_text)
    full_text = re.search(REGEX_SPAN_TEXT, full_text, re.IGNORECASE).group()
    full_text = full_text.replace("<TEXT>", "").replace("</TEXT>", "")
    full_text = full_text.strip("\n \t")
    return full_text


def split_sentences_timebank_extra(input_text: str) -> List[str]:
    full_sentences = re.findall(REGEX_SPAN_SENTENCE, input_text, re.IGNORECASE)
    for i in range(len(full_sentences)):
        full_sentences[i] = re.sub(r"<s>", "", full_sentences[i], flags=re.IGNORECASE)
        full_sentences[i] = (re.sub(r"</s>", "", full_sentences[i], flags=re.IGNORECASE)).strip("\n \t")
    return full_sentences

def generate_sentence_objects(full_sentences: List[str], makeinstance_objects: List[TagMakeinstance], tlink_objects: List[TagTlink], meta_time: TagDocumentMetaTime) -> List[Sentence]:

    sentences = []
    for index_in_document, full_sentence in enumerate(full_sentences):
        sentences += [_analyze_sentence_and_generate_targets(full_sentence, index_in_document, makeinstance_objects, tlink_objects, meta_time)]
    return sentences


def _construct_tag_regex(tag_name: str) -> str:
    """
    Takes name of a tag like EVENT (XML tag) and creates a pair
    of regex to match the opening and the end of the tag.
    Returns a pair of regexes.

    Example: EVENT -> Opening: <EVENT[^>]*?>, Closing: </EVENT>
    """
    regex_open_tag = r"<" + tag_name + r"[^>]*?>"
    regex_close_tag = r"</" + tag_name + r">"
    return regex_open_tag, regex_close_tag


def strip_article_content_timebank_extra(input_text: str) -> str:
    tagless_text = re.sub(REGEX_ANY_XMLTAG, "", input_text, flags=re.IGNORECASE)
    cleaned_text = tagless_text.strip("\n \t")
    cleaned_text = _clean_text(cleaned_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    cleaned_text = cleaned_text.strip("\n \t")
    return cleaned_text


def split_sentences_timebank_timeml(input_text: str) -> List[str]:
    sentences = SENTENCE_TOKENIZER.tokenize(input_text)
    #Repair sentences where tokenizer split inside a TIMEX3 tag
    for index in range(len(sentences) - 1, 0, -1):
        sentence = sentences[index]
        regex_timex3_start = r"<TIMEX3"
        regex_timex3_end = r"</TIMEX3>"
        uppercase_sentence = sentence.upper()
        timex3_start_tokens = re.findall(regex_timex3_start, uppercase_sentence, flags=re.IGNORECASE)
        timex3_end_tokens = re.findall(regex_timex3_end, uppercase_sentence, flags=re.IGNORECASE)

        #Check if the first timex3 token is a start token and not an end token
        is_first_timex_start_token_before_end = uppercase_sentence.strip() != "" and is_first_appearing_regex(regex_timex3_start, regex_timex3_end, sentence)

        if (len(timex3_start_tokens) != len(timex3_end_tokens) and (index > 0) and is_first_timex_start_token_before_end):
            print("Warning, found broken timex3 tag in sentence (TimeMl):\n" + sentence + "\n\n")
            sentences[index - 1] += " " + sentences[index]
            print(f"REPAIRED SENTENCE --- {sentences[index - 1]}\n")
            del sentences[index]
        elif (len(timex3_start_tokens) != len(timex3_end_tokens) and not is_first_timex_start_token_before_end):
            #This means that the first timex3 token is in the sentence after this, but it wasn't recognized as such and doesn't exist in input text.
            #Warn and delete sentence
            print("WARNING SENTENCE CANNOT BE REPAIRED! Found broken timex3 tag in sentence (TimeMl):\n\n" + sentence)
            print("DELETING SENTENCE!\n")
            del sentences[index]
    return sentences


def is_first_appearing_regex(regex1: str, regex2: str, sentence: str) -> bool:
    """
    Checks if regex1 appears before regex2 in the sentence.
    """
    #Check if regex1 appears before regex2
    pattern = re.compile(f'{regex1}.*{regex2}', re.IGNORECASE)
    after_another_match = pattern.search(sentence)

    #Or regex2 doesn't exist, but regex1 does
    regex1_exists = (re.search(regex1, sentence, re.IGNORECASE)) is None
    regex2_exists = (re.search(regex2, sentence, re.IGNORECASE)) is None
    dont_exist = regex1_exists and not regex2_exists
    return (after_another_match is not None) or dont_exist


def extract_article_content_aquaint(input_text: str) -> str:
    full_text = input_text.replace("\n", " ")
    full_text = re.sub(r"\s{2,}", " ", full_text)
    full_text = re.search(REGEX_SPAN_TEXT, full_text, re.IGNORECASE).group()
    full_text = full_text.replace("<TEXT>", "").replace("</TEXT>", "")
    full_text = full_text.strip("\n \t")
    return full_text
    

def strip_article_content_aquaint(input_text: str) -> str:
    tagless_text = re.sub(REGEX_ANY_XMLTAG, "", input_text, flags=re.IGNORECASE)
    cleaned_text = tagless_text.strip("\n \t")
    cleaned_text = _clean_text(cleaned_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    cleaned_text = cleaned_text.strip("\n \t")
    return cleaned_text


def split_sentences_aquaint(input_text: str) -> List[str]:
    sentences = SENTENCE_TOKENIZER.tokenize(input_text)
    #Repair sentences where tokenizer split inside a TIMEX3 tag
    for index in range(len(sentences) - 1, 0, -1):
        sentence = sentences[index]
        regex_timex3_start = r"<TIMEX3"
        regex_timex3_end = r"</TIMEX3>"
        uppercase_sentence = sentence.upper()
        timex3_start_tokens = re.findall(regex_timex3_start, uppercase_sentence, flags=re.IGNORECASE)
        timex3_end_tokens = re.findall(regex_timex3_end, uppercase_sentence, flags=re.IGNORECASE)

        #Check if the first timex3 token is a start token and not an end token
        is_first_timex_start_token_before_end = uppercase_sentence.strip() != "" and is_first_appearing_regex(regex_timex3_start, regex_timex3_end, sentence)

        if (len(timex3_start_tokens) != len(timex3_end_tokens) and (index > 0) and is_first_timex_start_token_before_end):
            print("Warning, found broken timex3 tag in sentence (TimeMl):\n" + sentence + "\n")
            sentences[index - 1] += " " + sentences[index]
            print(f"REPAIRED SENTENCE --- {sentences[index - 1]}\n\n")
            del sentences[index]
        elif (len(timex3_start_tokens) != len(timex3_end_tokens) and not is_first_timex_start_token_before_end):
            #This means that the first timex3 token is in the sentence after this, but it wasn't recognized as such and doesn't exist in input text.
            #Warn and delete sentence
            print("WARNING SENTENCE CANNOT BE REPAIRED! Found broken timex3 tag in sentence (TimeMl):\n" + sentence)
            print("DELETING SENTENCE!\n\n")
            del sentences[index]

    return sentences


######
    sentences = SENTENCE_TOKENIZER.tokenize(input_text)
    #Repair sentences where tokenizer split inside a TIMEX3 tag
    for index in range(len(sentences) - 1, 0, -1):
        sentence = sentences[index]
        regex_timex3_start = r"<TIMEX3"
        regex_timex3_end = r"</TIMEX3>"
        uppercase_sentence = sentence.upper()
        timex3_start_tokens = re.findall(regex_timex3_start, uppercase_sentence, flags=re.IGNORECASE)
        timex3_end_tokens = re.findall(regex_timex3_end, uppercase_sentence, flags=re.IGNORECASE)

        #Check if the first timex3 token is a start token and not an end token
        is_first_timex_start_token_before_end = uppercase_sentence.strip() != "" and is_first_appearing_regex(regex_timex3_start, regex_timex3_end, sentence)

        if (len(timex3_start_tokens) != len(timex3_end_tokens) and (index > 0) and is_first_timex_start_token_before_end):
            print("Warning, found broken timex3 tag in sentence (TimeMl):\n" + sentence + "\n")
            sentences[index - 1] += " " + sentences[index]
            print(f"REPAIRED SENTENCE --- {sentences[index - 1]}\n")
            del sentences[index]
        elif (len(timex3_start_tokens) != len(timex3_end_tokens) and not is_first_timex_start_token_before_end):
            #This means that the first timex3 token is in the sentence after this, but it wasn't recognized as such and doesn't exist in input text.
            #Warn and delete sentence
            print("WARNING SENTENCE CANNOT BE REPAIRED! Found broken timex3 tag in sentence (TimeMl):\n" + sentence)
            print("DELETING SENTENCE!\n")
            del sentences[index]

#####


def _clean_text(text: str) -> str:
    for pattern, replacement in TEXT_CLEANING_PATTERNS:
        text = re.sub(pattern, replacement, text)

    for tag in TAGS_TO_DELETE:
        if tag.upper() in text.upper():
            regex_open_tag, regex_close_tag = _construct_tag_regex(tag)
            text = re.sub(regex_open_tag, "", text, flags=re.IGNORECASE)
            text = re.sub(regex_close_tag, "", text, flags=re.IGNORECASE)

    return text


def _pre_clean_sentence(sentence):
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace("?", " ? ")
    sentence = sentence.replace("!", " ! ")
    sentence = re.sub(r"\s{2,}", " ", sentence)
    sentence = sentence.replace("'s", " 's")
    
    for tag in TAGS_TO_DELETE:
        if tag.upper() in sentence.upper():
            regex_open_tag, regex_close_tag = _construct_tag_regex(tag)
            sentence = re.sub(regex_open_tag, "", sentence, flags=re.IGNORECASE)
            sentence = re.sub(regex_close_tag, "", sentence, flags=re.IGNORECASE)

    sentence = sentence.replace("-", " - ") #TODO is this fine? does it lead to problems?

    return sentence


def _analyze_sentence_and_generate_targets(
    sentence: str,
    index_in_document: int,
    makeinstance_objects: List[TagMakeinstance],
    tlink_objects: List[TagTlink],
    meta_time: TagDocumentMetaTime
) -> List[str]:
    original_sentence = sentence
    sentence = _pre_clean_sentence(sentence)

    sentence = re.sub(REGEX_XML_START_TIMEX3, TIMEX3_SPECIAL_TOKEN_START + " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub(REGEX_XML_END_TIMEX3, " " + TIMEX3_SPECIAL_TOKEN_END, sentence, flags=re.IGNORECASE)
    sentence = re.sub(REGEX_XML_START_EVENT, EVENT_SPECIAL_TOKEN_START + " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub(REGEX_XML_END_EVENT, " " + EVENT_SPECIAL_TOKEN_END, sentence, flags=re.IGNORECASE)

    timex3_objects, full_event_objects = _extract_entities_from_sentence(original_sentence, makeinstance_objects)
    labeled_entities, tokens = _tokenize_and_position_entities(sentence, full_event_objects, timex3_objects)
    entity_relations, dct_relations, dpt_relations = _find_relations_in_sentence(timex3_objects, full_event_objects, tlink_objects, meta_time)

    sentence_without_special_tokens = sentence.replace(EVENT_SPECIAL_TOKEN_START + " ", "")
    sentence_without_special_tokens = sentence_without_special_tokens.replace(" " + EVENT_SPECIAL_TOKEN_END, "")
    sentence_without_special_tokens = sentence_without_special_tokens.replace(TIMEX3_SPECIAL_TOKEN_START + " ", "")
    sentence_without_special_tokens = sentence_without_special_tokens.replace(" " + TIMEX3_SPECIAL_TOKEN_END, "")

    #Create sentence object
    result_sentence = Sentence(
        text=sentence_without_special_tokens,
        tagged_text=original_sentence,
        sentence_in_document=index_in_document,
        tokens=tokens,
        timex3_in_sentence=timex3_objects,
        events_in_sentence=full_event_objects,
        entity_relations=entity_relations,
        dct_relations=dct_relations,
        dpt_relations=dpt_relations,
        labeled_entities=labeled_entities
    )

    return result_sentence


def _extract_entities_from_sentence(sentence: str, makeinstance_objects: List[TagMakeinstance]) -> tuple:
    timex3_objects = extract_timex3_objects(sentence)
    event_objects = extract_event_objects(sentence)
    event_make_instance_pairs = find_matching_pairs(event_objects, makeinstance_objects, "eid", "event_id")
    full_event_objects = []
    for event_object, makeinstance_object in event_make_instance_pairs:
        full_event_objects += [TagFullEvent(event_object, makeinstance_object)]

    return timex3_objects, full_event_objects


def _find_relations_in_sentence(
    timex3_objects: List[TagTimex3],
    full_event_objects: List[TagFullEvent],
    tlink_objects: List[TagTlink],
    meta_time: TagDocumentMetaTime
)-> tuple:
    temporal_objects = timex3_objects + full_event_objects
    if meta_time.dct_object:
        temporal_objects += [meta_time.dct_object]
    if meta_time.dpt_object:
        temporal_objects += [meta_time.dpt_object]
    entity_relations = []
    contains_dct = "-DCT"
    contains_dpt = "-DPT"
    for source_entity in temporal_objects:
        for target_entity in temporal_objects:
            if source_entity == target_entity:
                continue
            
            relation_type = ""
            if (isinstance(source_entity, TagTimex3) and source_entity.is_dct) or (isinstance(target_entity, TagTimex3) and target_entity.is_dct):
                relation_type += contains_dct
            if (isinstance(source_entity, TagTimex3) and source_entity.is_dpt) or (isinstance(target_entity, TagTimex3) and target_entity.is_dpt):
                relation_type += contains_dpt

            for tlink in tlink_objects:
                #First position in Tlink: eventInstanceID, timeID
                #Second position in Tlink: relatedToTime, relatedToEventInstance
                if isinstance(source_entity, TagTimex3):
                    if isinstance(target_entity, TagTimex3): #TT
                        if tlink.timeID == source_entity.tid and tlink.relatedToTime == target_entity.tid:
                            relation_type = "TT" + relation_type
                            entity_relations += [TemporalRelation(tlink.relType, source_entity, target_entity, tlink, relation_type)]
                    elif isinstance(target_entity, TagFullEvent): #TE
                        if tlink.timeID == source_entity.tid and tlink.relatedToEventInstance == target_entity.eiid:
                            relation_type = "TE" + relation_type
                            entity_relations += [TemporalRelation(tlink.relType, source_entity, target_entity, tlink, relation_type)]
                    else:
                        raise Exception("Unknown temporal object.")
                elif isinstance(source_entity, TagFullEvent):
                    if isinstance(target_entity, TagTimex3): #ET
                        if tlink.eventInstanceID == source_entity.eiid and tlink.relatedToTime == target_entity.tid:
                            relation_type = "ET" + relation_type
                            entity_relations += [TemporalRelation(tlink.relType, source_entity, target_entity, tlink, relation_type)]
                    elif isinstance(target_entity, TagFullEvent): #EE
                        if tlink.eventInstanceID == source_entity.eiid and tlink.relatedToEventInstance == target_entity.eiid:
                            relation_type = "EE" + relation_type
                            entity_relations += [TemporalRelation(tlink.relType, source_entity, target_entity, tlink, relation_type)]
                    else:
                        raise Exception("Unknown temporal object.")
    dct_relations = []
    dpt_relations = []
    for entity_relation in entity_relations:
        relation_code = entity_relation.relation_code
        if contains_dct in relation_code:
            dct_relations += [entity_relation]
        elif contains_dpt in relation_code:
            dpt_relations += [entity_relation]

    for entity_relation in entity_relations:
        relation_code = entity_relation.relation_code
        if contains_dct in relation_code or contains_dpt in relation_code:
            entity_relations.remove(entity_relation)

    return entity_relations, dct_relations, dpt_relations


def _tokenize_and_position_entities(sentence, full_event_objects, timex3_objects) -> tuple:
    #TODO maybe problem in here?
    sentence_with_special_tokens = sentence#.replace("-", " - ")
    tokens_with_special_tokens = WORD_TOKENIZER.tokenize(sentence_with_special_tokens)
    token_index = -1
    event_start = -1
    event_end = -1
    timex3_start = -1
    timex3_end = -1
    event_stack_pointer = 0
    timex3_stack_pointer = 0
    labeled_entities = []
    for token in tokens_with_special_tokens:
        token_compare = f" {token.upper().strip()} "
        if token_compare not in [EVENT_SPECIAL_TOKEN_START, EVENT_SPECIAL_TOKEN_END, TIMEX3_SPECIAL_TOKEN_START, TIMEX3_SPECIAL_TOKEN_END]:
            token_index += 1 #Skip special tokens
            continue
            
        if token_compare == EVENT_SPECIAL_TOKEN_START:
            event_start = token_index + 1
        elif token_compare == EVENT_SPECIAL_TOKEN_END:
            event_end = token_index
            full_event_objects[event_stack_pointer].set_position_in_sentence(event_start, event_end)
            corresponding_event = full_event_objects[event_stack_pointer]
            labeled_entities += [(event_start, event_end, corresponding_event)]
            event_stack_pointer += 1
        elif token_compare == TIMEX3_SPECIAL_TOKEN_START:
            timex3_start = token_index + 1
        elif token_compare == TIMEX3_SPECIAL_TOKEN_END:
            timex3_end = token_index
            timex3_objects[timex3_stack_pointer].set_position_in_sentence(timex3_start, timex3_end)
            corresponding_timex3 = timex3_objects[timex3_stack_pointer]
            labeled_entities += [(timex3_start, timex3_end, corresponding_timex3)]
            timex3_stack_pointer += 1

    tokens = []
    for token in tokens_with_special_tokens:
        if f" {token.upper().strip()} " not in [EVENT_SPECIAL_TOKEN_START, EVENT_SPECIAL_TOKEN_END, TIMEX3_SPECIAL_TOKEN_START, TIMEX3_SPECIAL_TOKEN_END]:
            tokens += [token]

    return labeled_entities, tokens


def create_tempeval_document(
        filepath: str,
        filecontents: str,
        article_contents: str,
        clean_text: str,
        sentences: List[Sentence],
        meta_time: TagDocumentMetaTime,
        full_event_objects: List[TagFullEvent],
        timex3_objects: List[TagTimex3],
        tlink_objects: List[TagTlink],
        makeinstance_objects: List[TagMakeinstance],
        signal_objects: List[TagSignal]
    ) -> tuple:
    filename = os.path.basename(filepath)

    intra_sentence_relations = [] #sum of the sentence relations, save sentence object
    dct_relations = [] #sum of the sentence dct relations, save sentence object
    dpt_relations = [] #sum of the sentence dpt relations, save sentence object
    for sentence in sentences:
        intra_sentence_relations += sentence.entity_relations
        dct_relations += sentence.dct_relations
        dpt_relations += sentence.dpt_relations

    inter_sentence_tlink_objects = copy.deepcopy(tlink_objects)
    number_of_removed = 0
    for relation in intra_sentence_relations:
        tlink_id = relation.tlink.lid
        for inter_sentence_tlink_object in inter_sentence_tlink_objects:
            if inter_sentence_tlink_object.lid == tlink_id:
                number_of_removed += 1
                inter_sentence_tlink_objects.remove(inter_sentence_tlink_object)

    for relation in dct_relations:
        tlink_id = relation.tlink.lid
        for inter_sentence_tlink_object in inter_sentence_tlink_objects:
            if inter_sentence_tlink_object.lid == tlink_id:
                number_of_removed += 1
                inter_sentence_tlink_objects.remove(inter_sentence_tlink_object)

    #TODO: Might be an error here
    for relation in dpt_relations:
        tlink_id = relation.tlink.lid
        for inter_sentence_tlink_object in inter_sentence_tlink_objects:
            if inter_sentence_tlink_object.lid == tlink_id:
                number_of_removed += 1
                inter_sentence_tlink_objects.remove(inter_sentence_tlink_object)
        
    #assert len(tlink_objects) == (len(inter_sentence_tlink_objects) + len(intra_sentence_relations) + len(dct_relations) + len(dpt_relations))
    document = Document(
        filename=filename,
        filepath=filepath,
        clean_text=clean_text,

        sentences=sentences,

        makeinstance_tags=makeinstance_objects,
        tlink_tags=tlink_objects,
        event_tags=full_event_objects,
        timex3_tags=timex3_objects,

        intra_sentence_relations=intra_sentence_relations,
        inter_sentence_relations=inter_sentence_tlink_objects,
        dct_relations=dct_relations,
        dpt_relations=dpt_relations,

        dct=meta_time.dct_object,
        dpt=meta_time.dpt_object
    )

    return document