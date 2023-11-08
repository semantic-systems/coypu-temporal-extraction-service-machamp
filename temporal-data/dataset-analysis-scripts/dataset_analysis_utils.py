from typing import List, Tuple

class AnalysisResult:
    """
    The AnalysisResult class is used to save information about datasets.
    """
    def __init__(self) -> None:
        self.text: str = "" #Contains the analysis results
        self.heading: str = "" #Contains the heading of the analysis results
        self.underline_character: str = "-"
        self.decorate_character_horizontal: str = "="
        self.decorate_character_vertical: str = "|"

    def set_heading(self, heading: str) -> None:
        """
        Sets the heading of the analysis results.

        Args:
            heading: The heading of the analysis results.
        """
        self.heading = self.add_big_heading(heading)

    def append(self, text_element: str) -> None:
        """
        Appends a text element to the analysis result text.

        Args:
            text_element: The text element to be appended.
        """
        self.text = self.text + "\n" + text_element

    def newline(self):
        """
        Adds a newline to the analysis result text.
        """
        self.text = self.text + "\n"
    
    def add_small_heading(self, small_heading_text: str, underline_character: str = "-"):
        """
        Adds a line of "underline_characters" under a text. Default is "-". This is used for small headings.

        Args:
            small_heading_text: The text to be decorated.
            underline_character: The character used for the underline.

        Returns:
            A decorated text.
        """
        text_length = len(small_heading_text)
        underline = "".join([underline_character for _ in range(text_length)])
        self.text = self.text + "\n" + small_heading_text + "\n" + underline
    
    def add_big_heading(self, text: str, decorate_character_horizontal: str = "=", decorate_character_vertical: str = "|") -> None:
        """
        Decorates a text with a line of "=" on top and bottom. This is used for big headings.

        Args:
            text: The text to be decorated.
            decorate_character_horizontal: The character used for the horizontal decoration.
            decorate_character_vertical: The character used for the vertical decoration.

        Returns:
            A decorated text.
        """
        text_length = len(text)
        object_length = text_length * 2
        padding_length = (text_length / 2) - 1

        top_bottom = "".join([decorate_character_horizontal for _ in range(object_length)])
        padding = "".join([" " for _ in range(int(padding_length))])
        middle = f"{decorate_character_vertical}{padding}{text}{padding}{decorate_character_vertical}"
        return top_bottom + "\n" + middle + "\n" + top_bottom
    
    def get_analysis_result(self) -> str:
        """
        Returns the analysis results.
        """
        return f"{self.heading}\n{self.text}" 