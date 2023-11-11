import json
import nltk
from analysis_utils.preprocessing_utils import DatasetNltkTokenizer
from collections import Counter
import re
from typing import List, Tuple
from analysis_utils.dataset_analysis_utils import AnalysisResult

class PateDatasetAnalyzer:
    """
    The PateDatasetAnalyzer is used to analyze the PATE dataset.

    The intended use is as follows:
    1. Create an instance of the PateDatasetAnalyzer.
    2. Call the analyze() method to analyze the dataset.
    3. Call the generate_analysis_results() method to generate the analysis results.
    4. (optional) Call the generate_analysis_data_entry_results() method to generate 
        the analysis results for each data entry.
    5. (optional) Call the generate_analysis_data_entry_timex3_empty_results() method to 
        generate the analysis results for each data entry that has an empty timex3 type.
    6. Call the save_analysis_results() method to save the analysis results to a file.

    Args:
        dataset_filepaths: A list of filepaths to the dataset files.
        analysis_output_path: The path to the analysis output file.
    """
    def __init__(self, dataset_filepaths: List[str], analysis_output_path: str) -> None:
        """"
        Constructor: Initializes the PateDatasetAnalyzer. The analyzer is used to analyze the PATE dataset.
        """
        #Load tokenizers
        def initiate_tokenizers() -> Tuple[DatasetNltkTokenizer, nltk.data.load]:
            """
            Initiates the tokenizers.

            Returns:
                A tuple containing the word tokenizer and sentence tokenizer.
            """
            sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            word_tokenizer = DatasetNltkTokenizer()
            return word_tokenizer, sentence_tokenizer
        
        self.word_tokenizer, self.sentence_tokenizer = initiate_tokenizers()


        #Load dataset
        def load_dataset(dataset_filepaths: List[str]) -> List[dict]:
            """
            Loads the dataset from a file.

            Args:
                dataset_filepaths: List of filepath to the dataset files. Tested only with fullfile paths.

            Returns:
                A list of dictionaries. Each dictionary represents a dataset instance.
            """
            dataset_instances = list()
            for dataset_filepath in dataset_filepaths:
                with open(dataset_filepath, 'r') as file:
                    dataset_instances += [json.load(file)]
            return dataset_instances
        
        self.dataset_filepaths = dataset_filepaths
        self.dataset_instances = load_dataset(dataset_filepaths)
        self.analysis_output_path = analysis_output_path

        #Special tokens that are not considered as word tokens. Needed to filter sets of tokens.
        self.special_tokens = ["", ",", ".", "!", "?", ";", ":", "'"]   


        #Initialize variables reuqired for analysis
        self.total_number_of_dataobjects = 0 #Total number of data objects in the dataset

        """
        Sentence related variables
        """
        self.total_number_of_sentences = 0 #Total number of sentences in the dataset
        self.total_number_of_0sentence_input = 0 #Total number of data-elements with sentence length 0
        self.total_number_of_1sentence_input = 0 #Total number of data-elements with sentence length 1
        self.total_number_of_2sentence_input = 0 #Total number of data-elements with sentence length 2
        self.total_number_of_3sentence_input = 0 #Total number of data-elements with sentence length 3
        self.total_number_of_4sentence_input = 0 #Total number of data-elements with sentence length 4
        self.total_number_of_larger_5sentence_input = 0 #Total number of data-elements with sentence length 5 and larger
        self.average_sentences_per_data_object = 0 #Average number of sentences per data object/dataset element
        self.number_of_sentences_that_contain_timex3 = 0 #Number of sentences that contain timex3 tags


        """
        Token related variables
        """
        self.total_number_of_tokens = 0 #Total number of tokens in the dataset
        self.total_number_of_word_tokens = 0 #Total number of word tokens in the dataset (all tokens except special tokens)
        self.average_tokens_per_data_object = 0 #Average number of tokens per data object/dataset element
        self.average_word_tokens_per_data_object = 0 #Average number of word tokens per data object/dataset element

        
        """
        Entity related variables
        """
        self.total_number_of_entities = 0 #Total number of entities in the dataset. Note that PATE dataset may have multiple timex tags in an entity
        self.total_number_of_entities_without_empty = 0 #Total number of entities in the dataset that are not empty i.e. consists solely of a single typed "empty" timex3 tag
        self.total_number_of_timex_tags = 0 #Total number of timex3 tags in the dataset
        self.total_tokens_length_of_filtered_timex_expressions = 0 #Filter out special tokens, the filterted tokens are also refered to as "word tokens"
        self.average_number_of_entities_per_data_object = 0 #Average number of entities per data object/dataset element
        self.total_number_of_entity_timex3_types = 0 #Total number of entity timex3 types in the dataset
        
        self.counter_entity_timex3_types = Counter() #Counts the timex3 types in the dataset. Types: [SET, DATE, TIME, DURATION, empty]
        self.set_of_concrete_timex3_tokens = set() #All timex3 tokens used in dataset e.g. "2019-01-01", "20:05", "today"


        """
        Saves a representation of the analysis results. The analysis results are saved as a string.
        """
        self.analysis_results: str = "" #Contains the summary of the analysis results for the whole dataset


        """
        Contains the analysis results for each data entry. Each AnalysisResult has a string representation
        """
        self.analysis_data_entry_results: List[AnalysisResult] = list() #Contains the analysis results for each data entry
        self.analysis_data_entry_results_text: str = "" #Contains the analysis results as a string for each data entry


        """
        Contains the analysis results for each data entry that exclusevly has an empty timex3 type. Each AnalysisResult has a string representation
        """
        self.analysis_data_entry_timex3_empty_results: List[AnalysisResult] = list() #Contains the analysis results for each data entry that has an empty timex3 type
        self.analysis_data_entry_timex3_empty_results_text: str = "" #Contains the analysis results as a string for each data entry that has an empty timex3 type


        """
        Vocabulary related variables
        """
        self.unique_timex_patterns = set() #Contains a set of unique timex patterns used in the dataset
        self.vocabulary = set() #Contains the whole vocabulary of the dataset
        self.vocabulary_for_timex3_date = set() #Contains the vocabulary for texts labeled with timex3 type "DATE"
        self.vocabulary_for_timex3_time = set() #Contains the vocabulary for texts labeled with timex3 type "TIME"
        self.vocabulary_for_timex3_duration = set() #Contains the vocabulary for texts labeled with timex3 type "DURATION"
        self.vocabulary_for_timex3_set = set() #Contains the vocabulary for texts labeled with timex3 type "SET"
        

        """
        Flags that indicate if the analysis results are generated
        """
        self.is_analysis_result_generated = False
        self.is_data_entry_result_generated = False
        self.is_data_entry_timex3_empty_results_generated = False


        """
        Formatting variables
        """
        self.newline_splitter = "\n" * 50
    

    def extract_from_dataset_entry(self, dataset_entry: dict) -> Tuple[List[str], List[dict]]:
        """
        Extracts the text and entities from a dataset entry.

        Args:
            dataset_entry: A dataset entry.

        Returns:
            A tuple containing the text and entities of the dataset entry.
        """
        text = dataset_entry["text"]
        entities = dataset_entry["entities"]
        return text, entities
    
    def filter_special_tokens(self, items) -> List[str]:
        """
        Filters special tokens from a list of tokens.

        Args:
            items: A list of tokens.

        Returns:
            A list of tokens without special tokens.
        """
        filtered_items = list()
        for item in items:
            if item.strip() not in self.special_tokens:
                filtered_items.append(item)
        return filtered_items
    
    def process_dataset_entities(self, entities: dict, original_text: str, skip_empty=False) -> List[dict]:
        """
        Processes the entities of a dataset entry. Identify empty entities and extract information about them.
        Recognize DURATION entities and reconstruct the original text from the beginPoint and endPoint.

        Args:
            entities: The entities of a dataset entry.

        Returns:
            A list of dictionaries. Each dictionary contains information about one entity.        
        """
        processed_entities = list()
        for entity in entities:
            timex_entities = entity["TIMEX3"]
            for timex3 in timex_entities:
                if skip_empty:
                    if timex3["expression"] == "" and timex3["beginPoint"] == "" and timex3["endPoint"] == "":
                        continue
                tokens = self.word_tokenizer.tokenize(timex3["expression"])
                timex3_type = timex3["type"].strip()
                timex3_type = timex3_type if timex3_type != "" else "empty"

                entity_information = dict()
                entity_information["type"] = timex3_type
                
                if timex3_type == "DURATION":
                    entity_information["duration_regex"] = self.pate_dataset_duration_span_regex_extractor(timex3["beginPoint"], timex3["endPoint"], entities)
                    pattern_matches = re.findall(entity_information["duration_regex"], original_text)
                    entity_information["text"] = pattern_matches[0] if len(pattern_matches) > 0 else ""
                else:
                    entity_information["duration_regex"] = ""
                    entity_information["text"] = timex3["expression"]

                entity_information["tokens"] = tokens
                entity_information["tokens_without_special_tokens"] = self.filter_special_tokens(tokens)

                processed_entities.append(entity_information)
        return processed_entities
    
    def process_dataset_entry(self, dataset_entry: dict) -> Tuple[str, List[str], List[str], List[str], List[dict], List[dict]]:
        """
        Processes a dataset entry.

        Args:
            dataset_entry: A dataset entry.

        Returns:
            A tuple containing the text, sentences, tokens, filtered tokens, entities and processed entities of the dataset entry.
        """
        text, entities = self.extract_from_dataset_entry(dataset_entry)
        sentences = self.sentence_tokenizer.tokenize(text)
        tokens = self.word_tokenizer.tokenize(text)
        filtered_tokens = self.filter_special_tokens(tokens)
        processed_entities = self.process_dataset_entities(entities, text)
        return text, sentences, tokens, filtered_tokens, entities, processed_entities
    
    def process_dataset_instance(self, dataset_instance: List[dict]) -> List[Tuple[str, List[str], List[str], List[str], List[dict], List[dict]]]:
        """
        Processes a dataset instance.

        Args:
            dataset_instance: A dataset instance.

        Returns:
            A list of tuples. Each tuple contains the text, sentences, tokens, filtered tokens, entities and processed entities of a dataset entry.
        """
        processed_instance = list()
        for dataset_entry in dataset_instance:
            text, sentences, tokens, filtered_tokens, entities, processed_entities = self.process_dataset_entry(dataset_entry)
            processed_instance.append((text, sentences, tokens, filtered_tokens, entities, processed_entities))
        return processed_instance
            
    
    def process_dataset_instances(self) -> List[List[Tuple[str, List[str], List[str], List[str], List[dict], List[dict]]]]:
        """
        Processes all dataset instances.

        Returns:
            A list of lists of tuples. Each tuple contains the text, sentences, tokens, filtered tokens, entities and processed entities of a dataset entry.
            The first list represents the dataset instances. The second list represents the dataset entries in each instance. The tuples represent the 
            information of a dataset entry.
        """
        processed_dataset_instances = list()
        for dataset_instance in self.dataset_instances:
            processed_instance = self.process_dataset_instance(dataset_instance)
            processed_dataset_instances.append(processed_instance)
        return processed_dataset_instances

    def pate_dataset_duration_span_regex_extractor(self, beginPoint: str, endPoint: str, entities: List[dict]) -> str:
        """
        Extracts the span regex for a duration timex3 tag.

        Args:
            beginPoint: The begin point of the duration timex3 tag.
            endPoint: The end point of the duration timex3 tag.
            entities: The entities of the dataset entry.

        Returns:
            A string containing the span regex for the duration timex3 tag.
        """
        span_regex = ".*"
        begin_regex = ""
        end_regex = ""

        for entity in entities:
            timex_entities = entity["TIMEX3"]
            for timex3 in timex_entities:
                if timex3["beginPoint"] == beginPoint:
                    for timex3 in timex_entities:
                        if timex3["tid"] == beginPoint:
                            begin_regex = timex3["expression"]
                if timex3["endPoint"] == endPoint:
                    for timex3 in timex_entities:
                        if timex3["tid"] == endPoint:
                            end_regex = timex3["expression"]
        return begin_regex + span_regex + end_regex

    def analyze(self) -> None:
        """
        Analyzes the dataset and saves the results into the class variables.
        """
        processed_dataset_instances = self.process_dataset_instances()
        for dataset_instance in processed_dataset_instances:
            for dataset_entry in dataset_instance:
                #Extract information from dataset entry
                text, sentences, tokens, filtered_tokens, entities, processed_entities = dataset_entry


                #Calculate the vocabulary
                for token in filtered_tokens:
                    self.vocabulary.add(token.lower())


                #Calculate how many words are used for each timex3 type
                for processed_entity in processed_entities:
                    timex3_type = processed_entity["type"].lower()
                    timex3_tokens_without_special_tokens = processed_entity["tokens_without_special_tokens"]
                    if timex3_type == "date":
                        for token in timex3_tokens_without_special_tokens:
                            self.vocabulary_for_timex3_date.add(token.lower())
                    elif timex3_type == "time":
                        for token in timex3_tokens_without_special_tokens:
                            self.vocabulary_for_timex3_time.add(token.lower())
                    elif timex3_type == "duration":
                        for token in timex3_tokens_without_special_tokens:
                            self.vocabulary_for_timex3_duration.add(token.lower())
                    elif timex3_type == "set":
                        for token in timex3_tokens_without_special_tokens:
                            self.vocabulary_for_timex3_set.add(token.lower())


                #Check how many sentences contain timex3 tags
                for sentence in sentences:
                    contains_timex3: bool = False
                    for processed_entity in processed_entities:
                        if processed_entity["text"] in sentence:
                            contains_timex3 = True
                    if contains_timex3:
                        self.number_of_sentences_that_contain_timex3 += 1


                #Create analysis results for each data entry
                data_entry_result = AnalysisResult()
                data_entry_result.append(f"Text: {text}")
                data_entry_result.append(f"Number of sentences: {len(sentences)}")
                data_entry_result.append(f"Number of tokens: {len(tokens)}")
                data_entry_result.append(f"Tokens: {tokens}")
                data_entry_result.append(f"Number of filtered tokens: {len(filtered_tokens)}")
                data_entry_result.append(f"Filtered tokens: {filtered_tokens}")
                data_entry_result.newline()
                data_entry_result.append(f"Number of entities: {len(processed_entities)}")
                for i, processed_entity in enumerate(processed_entities, start=1):
                    # Example (maybe structure like this in the future):
                    # Number of entities: 2
                    # (1)  Expression: '6:30 pm'; Type: 'TIME'; BeginPoint: ''; EndPoint: ''; TID: 't157'
                    #     Value: '2020-10-02T18:30'; AnchorTimeID: 't156'
                    # (2)  Expression: '6:30 pm'; Type: 'TIME'; BeginPoint: ''; EndPoint: ''; TID: 't157'
                    #     Value: '2020-10-02T18:30'; AnchorTimeID: 't156'

                    data_entry_result.append(f"({i})   Expression: '{processed_entity['text']}'; Type: '{processed_entity['type']}'")

                data_entry_result.newline()
                data_entry_result.newline()
                self.analysis_data_entry_results.append(data_entry_result) #Append to list of analysis results for each data entry


                # Check sentence length and save the number of sentences per data object
                self.total_number_of_sentences += len(sentences)
                if len(sentences) == 0:
                    self.total_number_of_0sentence_input += 1
                elif len(sentences) == 1:
                    self.total_number_of_1sentence_input += 1
                elif len(sentences) == 2:
                    self.total_number_of_2sentence_input += 1
                elif len(sentences) == 3:
                    self.total_number_of_3sentence_input += 1
                elif len(sentences) == 4:
                    self.total_number_of_4sentence_input += 1
                elif len(sentences) > 4:
                    self.total_number_of_larger_5sentence_input += 1


                #Process empty entities and add them to the results list
                for i, processed_entity in enumerate(processed_entities):
                    timex3_type = processed_entity["type"]
                    timex3_tokens_without_special_tokens = processed_entity["tokens_without_special_tokens"]
                    self.counter_entity_timex3_types.update({timex3_type})
                    self.set_of_concrete_timex3_tokens.update(timex3_tokens_without_special_tokens)
                    if timex3_type == "empty":
                        self.analysis_data_entry_timex3_empty_results += [data_entry_result]
                    else:
                        self.total_number_of_timex_tags += 1

                
                #Increment length counters
                self.total_number_of_dataobjects += 1
                self.total_number_of_tokens += len(tokens)
                self.total_number_of_word_tokens += len(filtered_tokens)

                #Note that PATE dataset may have multiple timex in an entity
                #That means that the number of entities is not equal to the number of timex3 tags used
                #Timex3 > Entities
                self.total_number_of_entities += len(entities) 


            #Calculate averages
            self.average_sentences_per_data_object = self.total_number_of_sentences / self.total_number_of_dataobjects
            self.average_number_of_tokens_per_sentence = self.total_number_of_tokens / self.total_number_of_sentences
            self.average_tokens_per_data_object = self.total_number_of_tokens / self.total_number_of_dataobjects
            self.average_word_tokens_per_data_object = self.total_number_of_word_tokens / self.total_number_of_dataobjects
            self.average_number_of_entities_per_data_object = self.total_number_of_entities / self.total_number_of_dataobjects


            #Calculate totals
            self.total_number_of_entity_timex3_types = len(self.counter_entity_timex3_types)
            self.total_tokens_length_of_filtered_timex_expressions += len(self.set_of_concrete_timex3_tokens)
        
            def replace_digits_with_placeholder(text, placeholder="<d>"):
                """
                Function to replace digits with a generic placeholder.

                Args:
                    text: The text to be processed.

                Returns:
                    A string with digits replaced by a generic placeholder.
                """
                result = re.sub(r"\d", placeholder, text)
                return result
            
            #Add to the set of unique timex patterns, all numbers are treated as the same
            self.unique_timex_patterns = set([replace_digits_with_placeholder(item) for item in self.set_of_concrete_timex3_tokens])

    def newline(self, text):
        """
        Adds a newline to a text.
        """
        return text + "\n"
    
    def underline(self, text: str) -> str:
        """
        Adds a line of "-" under a text.
        """
        text_length: int = len(text)
        underline: str = "".join(["-" for _ in range(text_length)])
        return text + "\n" + underline + "\n"

    def generate_analysis_results(self, print_vocabulary: bool = True):
        """
        Generates the analysis results.
        """
        ar = AnalysisResult()
        ar.set_heading("ANALYSIS RESULTS")
        ar.add_small_heading("Dataset General Information")
        ar.append(f"Number of dataset input files: {len(self.dataset_filepaths)}")
        ar.append(f"Dataset filepaths: {'; '.join([filepath for filepath in self.dataset_filepaths])}")
        ar.append(f"Dataset size: {self.total_number_of_dataobjects}")
        ar.append(f"Vocabulary size: {len(self.vocabulary)}")
        if print_vocabulary: ar.append(f"Vocabulary: {self.vocabulary}")
        ar.newline()
        ar.append(f"Vocabulary size for timex3 date: {len(self.vocabulary_for_timex3_date)}")
        ar.append(f"Vocabulary for timex3 date: {self.vocabulary_for_timex3_date}")
        ar.newline()
        ar.append(f"Vocabulary size for timex3 time: {len(self.vocabulary_for_timex3_time)}")
        ar.append(f"Vocabulary for timex3 time: {self.vocabulary_for_timex3_time}")
        ar.newline()
        ar.append(f"Vocabulary size for timex3 duration: {len(self.vocabulary_for_timex3_duration)}")
        ar.append(f"Vocabulary for timex3 duration: {self.vocabulary_for_timex3_duration}")
        ar.newline()
        ar.append(f"Vocabulary size for timex3 set: {len(self.vocabulary_for_timex3_set)}")
        ar.append(f"Vocabulary for timex3 set: {self.vocabulary_for_timex3_set}")
        ar.newline()
        ar.newline()
        ar.add_small_heading("Dataset Sentence Information")
        ar.append(f"Total number of sentences: {self.total_number_of_sentences}")
        ar.append(f"Number of sentences that contain timex3 tags {self.number_of_sentences_that_contain_timex3} ({(self.number_of_sentences_that_contain_timex3 / self.total_number_of_sentences) * 100:.2f}%)")
        ar.append(f"Total number of data objects with sentence length 0: {self.total_number_of_0sentence_input}")
        ar.append(f"Total number of data objects with sentence length 1: {self.total_number_of_1sentence_input}")
        ar.append(f"Total number of data objects with sentence length 2: {self.total_number_of_2sentence_input}")
        ar.append(f"Total number of data objects with sentence length 3: {self.total_number_of_3sentence_input}")
        ar.append(f"Total number of data objects with sentence length 4: {self.total_number_of_4sentence_input}")
        ar.append(f"Total number of data objects with sentence length 5 and larger {self.total_number_of_larger_5sentence_input}")
        ar.append(f"Average number of tokens per sentence: {self.average_number_of_tokens_per_sentence}")
        ar.newline()
        ar.append(f"Total number of tokens: {self.total_number_of_tokens}")
        ar.append(f"Total number of word tokens: {self.total_number_of_word_tokens}")
        ar.append(f"Average sentences per data object: {self.average_sentences_per_data_object}")
        ar.append(f"Average tokens per data object: {self.average_tokens_per_data_object}")
        ar.append(f"Average word tokens per data object: {self.average_word_tokens_per_data_object}")
        ar.newline()
        ar.newline()
        ar.add_small_heading("Dataset Entity Information")
        ar.append(f"Total number of entities: {self.total_number_of_entities}")
        ar.append(f"Total number of timex tags: {self.total_number_of_timex_tags}")
        ar.append(f"Total number of entity objects (excluding \"empty\" timex3 typed tags): {self.total_number_of_entities_without_empty}")
        ar.append(f"Average number of entities per data object: {self.average_number_of_entities_per_data_object}")
        ar.append(f"Total number of entity types: {self.total_number_of_entity_timex3_types}")
        ar.append(f"Entity types counter: {self.counter_entity_timex3_types}")
        #Calculate percentages
        for item, count in self.counter_entity_timex3_types.items():
            percentage = (count / self.total_number_of_timex_tags) * 100
            ar.append(f"{item}: {percentage:.2f}%")
        ar.append(f"Total number of entity word tokens: {self.total_tokens_length_of_filtered_timex_expressions}")
        ar.newline()
        ar.append(f"Set of concrete timex3 tokens: {self.set_of_concrete_timex3_tokens}")
        ar.newline()
        ar.append(f"Unique timex patterns: {self.unique_timex_patterns}")

        self.analysis_results = ar.get_analysis_result()
        self.is_analysis_result_generated = True

    def generate_analysis_data_entry_results(self):
        """
        Generates the analysis results for each data entry.
        """
        ar = AnalysisResult()
        ar.set_heading("ANALYSIS RESULTS FOR EACH DATA ENTRY")
        self.analysis_data_entry_results_text += ar.get_analysis_result()

        for data_entry in self.analysis_data_entry_results:
            self.analysis_data_entry_results_text += data_entry.get_analysis_result()
        self.is_data_entry_result_generated = True

    def generate_analysis_data_entry_timex3_empty_results(self):
        """
        Generates the analysis results for each data entry that has an empty timex3 type.
        """
        ar = AnalysisResult()
        ar.set_heading("ANALYSIS RESULTS FOR EACH DATA ENTRY THAT HAS AN EMPTY TIMEX3 TYPE")
        self.analysis_data_entry_timex3_empty_results_text += ar.get_analysis_result()

        for data_entry in self.analysis_data_entry_timex3_empty_results:
            self.analysis_data_entry_timex3_empty_results_text += data_entry.get_analysis_result()
        self.is_data_entry_timex3_empty_results_generated = True

    def decorate(self, text):
        """
        Decorates a text with a line of "=" on top and bottom.
        """
        text_length = len(text)
        object_length = text_length * 2
        padding_length = (text_length / 2) - 1

        top_bottom = "".join(["=" for _ in range(object_length)])
        padding = "".join([" " for _ in range(int(padding_length))])
        middle = f"|{padding}{text}{padding}|"
        return top_bottom + "\n" + middle + "\n" + top_bottom
    
    def save_analysis_results(self):
        """
        Saves the analysis results to a file.
        """
        with open(self.analysis_output_path, "w") as file:
            print(f"Output file is set to: {self.analysis_output_path}")
            if self.is_analysis_result_generated:
                file.write(self.analysis_results)
                file.write(self.newline_splitter)
                print("Writing analysis summary to file...")
            if self.is_data_entry_result_generated:
                file.write(self.analysis_data_entry_results_text)
                file.write(self.newline_splitter)
                print("Writing analysis results for each data entry to file...")
            if self.is_data_entry_timex3_empty_results_generated:
                file.write(self.analysis_data_entry_timex3_empty_results_text)
                print("Writing analysis results for each data entry that has an empty timex3 type to file...")
            

def main():
    #Path to original dataset files (not converted)
    pate_dataset_paths = ["/export/home/4kirsano/uie/dataset_processing/data/my_datasets/original/pate_and_snips/pate.json"]
    pate_analysis_output_path = "/export/home/4kirsano/uie/dataset_processing/scripts/myscripts/pate_dataset_analysis_output.txt"

    pate_dataset_analyzer = PateDatasetAnalyzer(pate_dataset_paths, pate_analysis_output_path)
    pate_dataset_analyzer.analyze()
    pate_dataset_analyzer.generate_analysis_results()
    pate_dataset_analyzer.generate_analysis_data_entry_results()
    pate_dataset_analyzer.generate_analysis_data_entry_timex3_empty_results()
    pate_dataset_analyzer.save_analysis_results()

if __name__ == "__main__":
    main()