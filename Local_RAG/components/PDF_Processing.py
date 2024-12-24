import fitz
from tqdm import tqdm
from spacy.lang.en import English
import re
import random
from pprint import pprint


class PDF_Processor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    @staticmethod
    def text_formatter(text: str) -> str:
        cleaned_text = text.replace("\n", " ").strip()
        return cleaned_text

    @staticmethod
    def split_list(input_list: list, slice_size: int) -> list[list[str]]:

        return [
            input_list[i : i + slice_size]
            for i in range(0, len(input_list), slice_size)
        ]

    def _read_PDF(self) -> list[dict]:
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
        nlp = English()

        nlp.add_pipe("sentencizer")
        for item in tqdm(pages_and_texts, desc="text to sentence"):
            item["sentences"] = list(nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])

        return pages_and_texts

    def _chunk_sentence(self, pages_and_texts: list, chunk_size: int = 10):
        for item in tqdm(pages_and_texts, desc="sentence to chunk"):
            item["sentence_chunks"] = self.split_list(item["sentences"], chunk_size)
            item["page_chunk_count"] = len(item["sentence_chunks"])

        return pages_and_texts

    def _pages_and_chunks(self, pages_and_texts: list):
        pages_and_chunks = []
        for item in tqdm(pages_and_texts, desc="splitting each chunk into its own"):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                joined_sentence_chunk = (
                    "".join(sentence_chunk).replace("  ", " ").strip()
                )
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

    def _remove_irrevalant_chunks(self, pages_and_chunks: list):
        revelant_pages_and_chunks = [
            item for item in pages_and_chunks if item["chunk_token_count"] > 30
        ]

        return revelant_pages_and_chunks

    def run(self):
        pages_and_texts = self._read_PDF()
        self._split_sentence(pages_and_texts)
        self._chunk_sentence(pages_and_texts)
        pages_and_chunks = self._pages_and_chunks(pages_and_texts)
        revelant_pages_and_chunks = self._remove_irrevalant_chunks(pages_and_chunks)
        return revelant_pages_and_chunks


if __name__ == "__main__":
    pdf_path = "human-nutrition-text.pdf"
    pdf_processor = PDF_Processor(pdf_path)
    pdf_results = pdf_processor.run()
    pprint(random.sample(pdf_results, k=2))
