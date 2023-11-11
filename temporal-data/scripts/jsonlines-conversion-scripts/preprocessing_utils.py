from typing import List, Tuple
import math
import nltk
from nltk.tokenize import word_tokenize

class DatasetNltkTokenizer:
    """
    Preprocessing tokenizer for UIE datasets.
    """
    def __init__(self) -> None:
        self.sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    def tokenize(self, text) -> List[str]:
        """
        Tokenizes given text into a list of individual tokens.

        Args:
            text (str): The input sentence to be tokenized.

        Returns:
            list: A list of tokens extracted from the sentence.

        Example:
            tokenizer.tokenize_sentences("The dog is hungry. The cat too.")
            => ["The", "dog", "is, "hungry", ".", "The", "cat", "too", "."]
        """
        tokens = list()
        sentences = self.sent_tokenizer.tokenize(text)
        for sentence in sentences:
            sentence_tokens = word_tokenize(sentence)
            for sentence_token in sentence_tokens:
                tokens.append(sentence_token)
        return tokens
    
    def sentence_tokenize(self, text) -> List[str]:
        return nltk.sent_tokenize(text)
    
    
def initiate_tokenizers() -> Tuple[DatasetNltkTokenizer, nltk.data.load]:
    """
    Initiates the tokenizers.

    Returns:
        A tuple containing the word tokenizer and sentence tokenizer.
    """
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    word_tokenizer = DatasetNltkTokenizer()
    return word_tokenizer, sentence_tokenizer


def slice_list(input_list:List, *slicers:float, debug:bool=False) -> List[List]:
    """
    Slice a list based on the provided slicers. The sum of all slicers should be 1.

    Args:
        input_list (List[str]): The input list to be sliced.
        *slicers (int): Variable number of integer slicers used to slice the input list. Sum of all slicers should be 1.

    Returns:
        List[List]: A list of lists containing the sliced portions of the input list.

    Example:
        slices_list([0,1,2,3,4,5,6,7,8,9], 0.5, 0.4, 0.1)
        Returns a List of 3 Lists:
        => [[0,1,2,3,4],[5,6,7,8],[9]]
    """
    size = len(input_list)
    slicers_amount = len(slicers)
    
    output_list = []
    low = 0
    for i, slicer in enumerate(slicers):
        high = int(math.floor(slicer * size) + low)
        if i == (slicers_amount-1):
            high = size
        sliced_list = input_list[low:high]
        if debug: print("Current slice:", sliced_list)
        low = high
        output_list += [sliced_list]
    if debug:
        print("Output is:")
        for sublist in output_list:
            print(sublist)
    return output_list


def slice_list_in_equal_parts(input_list:List, folds:int, debug:bool=False) -> List[List]:
    """
    Slice a list based on the provided slicers. The sum of all slicers should be 1.

    Args:
        input_list (List[str]): The input list to be sliced.
        *slicers (int): Variable number of integer slicers used to slice the input list. Sum of all slicers should be 1.

    Returns:
        List[List]: A list of lists containing the sliced portions of the input list.

    Example:
        slices_list([0,1,2,3,4,5,6,7,8,9], 0.5, 0.4, 0.1)
        Returns a List of 3 Lists:
        => [[0,1,2,3,4],[5,6,7,8],[9]]
    """
    size = len(input_list)
    slicer:float = 1.0 / folds
    output_list = []
    low = 0
    for i in range(folds):
        high = int(math.floor(slicer * size) + low)
        if i == (folds - 1):
            high = size
        sliced_list = input_list[low:high]
        if debug: print("Current slice:", sliced_list)
        low = high
        output_list += [sliced_list]
    if debug:
        print("Output is:")
        for sublist in output_list:
            print(sublist)
    return output_list


def find_sublist_in_list(input_list: List, sublist: List) -> Tuple[int, int]:
    """
    Find a sublist within a given list and return the start and end indices of the found sublist.

    Args:
        input_list (List): The list to search within.
        sublist (List): The sublist to find within the input list.

    Returns:
        tuple: A tuple containing the start and end indices of the found sublist in the input list.
               If the sublist is not found, (-1, -1) is returned.

    Example:
        find_sublist_in_list([0,1,2,3,4,5], [2,3,4]) => (2,4)
        find_sublist_in_list([0,1,2,3,4,5], [2,9,9]) => (-1,-1)
    """
    sublist_size = len(sublist)
    sublist_start_index = -1
    sublist_end_index = -1
    lower_sublist = [x.lower().strip() for x in sublist]

    #Check all sublists of "input_list" of the same length as "sublist"
    for i in range(len(input_list) - sublist_size + 1):
        slice = input_list[i:(i+sublist_size)]

        if [x.lower().strip() for x in slice] == lower_sublist:
            sublist_start_index = i
            sublist_end_index = i + sublist_size - 1
    return (sublist_start_index, sublist_end_index)