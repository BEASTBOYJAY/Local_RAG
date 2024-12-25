import fitz
from tqdm import tqdm
from spacy.lang.en import English
import re
import random
from pprint import pprint


class PDF_Processor:
    """
    A class to process PDF documents by extracting and formatting text,
    splitting sentences into chunks, and filtering irrelevant content.

    Attributes:
        pdf_path (str): The path to the PDF file to be processed.
    """

    def __init__(self, pdf_path):
        """
        Initializes the PDF_Processor with the given PDF file path.

        Args:
            pdf_path (str): Path to the PDF file to be processed.
        """
        self.pdf_path = pdf_path

    @staticmethod
    def text_formatter(text: str) -> str:
        """
        Cleans and formats the extracted text by removing newlines and stripping extra spaces.

        Args:
            text (str): The raw text extracted from a PDF page.

        Returns:
            str: The formatted text with newlines removed and excess spaces trimmed.
        """
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    @staticmethod
    def split_list(input_list: list, slice_size: int) -> list[list[str]]:
        """
        Splits a list into smaller sublists of a specified size.

        Args:
            input_list (list): The list to be split into chunks.
            slice_size (int): The size of each chunk.

        Returns:
            list[list[str]]: A list containing sublists of the specified size.
        """
        return [
            input_list[i : i + slice_size]
            for i in range(0, len(input_list), slice_size)
        ]

    def _read_PDF(self) -> list[dict]:
        """
        Reads the PDF document, extracts the text, and calculates various metrics for each page.

        Returns:
            list[dict]: A list of dictionaries containing extracted text and page statistics.
        """
        try:
            pdf_document = fitz.open(self.pdf_path)
        except fitz.FileDataError:
            print(f"Error: Unable to open PDF file '{self.pdf_path}'.")

        pages_and_texts = []
        for page_number, page in tqdm(
            enumerate(pdf_document), total=len(pdf_document), desc="Reading PDF"
        ):
            text = page.get_text()
            text = self.text_formatter(text)
            pages_and_texts.append(
                {
                    "page_number": page_number,
                    "page_char_count": len(text),
                    "page_word_count": len(text.split(" ")),
                    "page_sentence_count_raw": len(text.split(". ")),
                    "page_token_count": len(text) / 4,
                    "text": text,
                }
            )
        return pages_and_texts

    def _split_sentence(self, pages_and_texts: list):
        """
        Splits the extracted text into individual sentences using SpaCy's sentence boundary detection.

        Args:
            pages_and_texts (list[dict]): A list of dictionaries containing page-level text data.

        Returns:
            list[dict]: The updated list with sentence-level data for each page.
        """
        nlp = English()
        nlp.add_pipe("sentencizer")
        for item in tqdm(pages_and_texts, desc="Text to sentence"):
            item["sentences"] = list(nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])

        return pages_and_texts

    def _chunk_sentence(self, pages_and_texts: list, chunk_size: int = 10):
        """
        Breaks down the sentences into smaller chunks for easier processing.

        Args:
            pages_and_texts (list[dict]): A list of dictionaries containing sentence-level data.
            chunk_size (int): The number of sentences per chunk (default is 10).

        Returns:
            list[dict]: The updated list with sentence chunks for each page.
        """
        for item in tqdm(pages_and_texts, desc="Sentence to chunk"):
            item["sentence_chunks"] = self.split_list(item["sentences"], chunk_size)
            item["page_chunk_count"] = len(item["sentence_chunks"])

        return pages_and_texts

    def _pages_and_chunks(self, pages_and_texts: list):
        """
        Converts sentence chunks into dictionaries with additional metadata (e.g., character count, token count).

        Args:
            pages_and_texts (list[dict]): A list of dictionaries containing sentence chunks.

        Returns:
            list[dict]: A list of dictionaries where each dictionary represents a chunk with metadata.
        """
        pages_and_chunks = []
        for item in tqdm(pages_and_texts, desc="Splitting each chunk into its own"):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentence chunks into a single string and clean up any excess spaces.
                joined_sentence_chunk = (
                    "".join(sentence_chunk).replace("  ", " ").strip()
                )
                # Fix any missing spaces after periods (e.g., "Hello.World" becomes "Hello. World").
                joined_sentence_chunk = re.sub(
                    r"\.([A-Z])", r". \1", joined_sentence_chunk
                )
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len(
                    [word for word in joined_sentence_chunk.split(" ")]
                )
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

                pages_and_chunks.append(chunk_dict)

        return pages_and_chunks

    def _remove_irrelevant_chunks(self, pages_and_chunks: list):
        """
        Filters out chunks that are too small based on token count (threshold is set to 30 tokens).

        Args:
            pages_and_chunks (list[dict]): A list of dictionaries containing chunk-level data.

        Returns:
            list[dict]: A list of relevant chunks (with token count > 30).
        """
        relevant_pages_and_chunks = [
            item for item in pages_and_chunks if item["chunk_token_count"] > 30
        ]

        return relevant_pages_and_chunks

    def run(self):
        """
        Executes the PDF processing pipeline, which includes reading the PDF, splitting text into sentences,
        chunking sentences, and filtering out irrelevant chunks.

        Returns:
            list[dict]: A list of relevant sentence chunks with metadata.
        """
        pages_and_texts = self._read_PDF()  # Read the PDF and extract text.
        self._split_sentence(pages_and_texts)  # Split text into sentences.
        self._chunk_sentence(pages_and_texts)  # Chunk sentences into smaller sections.
        pages_and_chunks = self._pages_and_chunks(
            pages_and_texts
        )  # Create chunks with metadata.
        relevant_pages_and_chunks = self._remove_irrelevant_chunks(
            pages_and_chunks
        )  # Filter out small chunks.
        return relevant_pages_and_chunks


if __name__ == "__main__":
    # Instantiate the PDF_Processor with the path to the PDF document.
    pdf_processor = PDF_Processor(pdf_path="document.pdf")
    # Run the PDF processing pipeline and get relevant sentence chunks.
    result = pdf_processor.run()
    pprint(result)
