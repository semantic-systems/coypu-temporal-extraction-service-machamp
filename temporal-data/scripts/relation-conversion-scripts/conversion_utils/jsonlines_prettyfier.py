import json 
from typing import List

class JSONLinesPrettifier:
    """
    Converts a jsonlines file to a file with pretty formatting.
    """
    def __init__(self, input_jsonlines_filepaths: List[str] = list(), output_filepaths: List[str] = list()):
        if len(input_jsonlines_filepaths) != len(output_filepaths):
            raise Exception("The number of input files must be equal to the number of output files.")
        self.input_jsonlines_filepaths = input_jsonlines_filepaths
        self.output_filepaths = output_filepaths
        self.separator = "\n\n" + ("-" * 100) + "\n\n"

    def append_to_inputfiles(self, input_filepath: str) -> None:
        """
        Appends a filepath to the input filepaths.
        """
        self.input_jsonlines_filepaths.append(input_filepath)

    def append_to_outputfiles(self, output_filepath: str) -> None:
        """
        Appends a filepath to the output filepaths.
        """
        self.output_filepaths.append(output_filepath)

    def run(self):
        """
        Runs the conversion.
        """
        for i in range(len(self.input_jsonlines_filepaths)):
            current_input_filepath = self.input_jsonlines_filepaths[i]
            current_output_filepath = self.output_filepaths[i]
            self.convert_jsonlines_to_json(current_input_filepath, current_output_filepath)

    def prettify_json(self, json_line: str) -> str:
        """
        Prettifies a json line.

        Args:
            json_line (str): A json line. Represents a json object.

        Returns:
            str: A prettified json object.
        """
        parsed_json = json.loads(json_line)
        prettified_json = json.dumps(parsed_json, indent=4)
        return prettified_json

    def convert_jsonlines_to_json(self, input_file: str, output_file:str) -> None:
        """
        Converts a jsonlines file to a json file with pretty formatting.
        """
        with open(input_file, "r") as input_file, open(output_file, "w") as output_file:
            for line in input_file:
                prettified_json = self.prettify_json(line)
                output_file.write(prettified_json + self.separator)
