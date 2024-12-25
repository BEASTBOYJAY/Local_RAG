from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from PDF_Processing import PDF_Processor
import pandas as pd


class SaveEmbeddings:
    """
    A class to generate and save sentence embeddings from a PDF document.

    This class processes a PDF file, extracts text chunks, generates embeddings
    for each chunk using a SentenceTransformer model, and saves the results to a CSV file.

    Attributes:
        device (str): The device to run the embedding model on ("cuda" if GPU is available, else "cpu").
        pdf_processor (PDF_Processor): An instance of PDF_Processor to extract text chunks from the PDF.
        pages_and_chunks (list[dict]): A list of dictionaries containing page-wise text chunks.
        embedding_model (SentenceTransformer): Pre-trained sentence transformer model for generating embeddings.
    """

    def __init__(self, pdf_path, embedding_model="all-mpnet-base-v2"):
        """
        Initializes the SaveEmbeddings class with the specified PDF file and embedding model.

        Args:
            pdf_path (str): Path to the PDF file to be processed.
            embedding_model (str): Name or path of the pre-trained embedding model. Defaults to "all-mpnet-base-v2".
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the PDF processor to extract text chunks from the PDF.
        self.pdf_processor = PDF_Processor(pdf_path=pdf_path)
        # Process the PDF and extract page-wise text chunks.
        self.pages_and_chunks = self.pdf_processor.run()
        # Load the sentence transformer model.
        self.embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model, device=self.device
        )

    def _generate_embeddings(self):
        """
        Generates sentence embeddings for the extracted text chunks.

        Iterates over the pages and chunks extracted from the PDF and adds an
        "embedding" key to each item in the `pages_and_chunks` list.
        """
        for item in tqdm(self.pages_and_chunks, desc="Generating embeddings"):
            # Generate embeddings for the sentence chunk and add it to the item.
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])

    def _save_embeddings(self):
        """
        Saves the extracted chunks and their embeddings to a CSV file.

        The file is saved as "embeddings.csv" in the current working directory.
        """
        # Convert the list of dictionaries to a DataFrame for saving as CSV.
        data_frame = pd.DataFrame(self.pages_and_chunks)
        data_frame.to_csv("embeddings.csv", index=False)

    def run(self):
        """
        Executes the embedding generation and saving process.

        This method calls the private `_generate_embeddings` and `_save_embeddings`
        methods sequentially to complete the process.
        """
        self._generate_embeddings()  # Generate embeddings for text chunks.
        self._save_embeddings()  # Save the embeddings to a CSV file.


if __name__ == "__main__":
    # Create an instance of SaveEmbeddings with the specified PDF file path.
    save_embeddings = SaveEmbeddings(pdf_path="human-nutrition-text.pdf")
    # Run the embedding generation and saving process.
    save_embeddings.run()
