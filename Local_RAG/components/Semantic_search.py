import numpy as np
import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer


class Semantic_search:
    """
    A class to perform semantic search on pre-computed embeddings of text data (e.g., from a PDF).
    This class loads the embeddings, processes the embeddings, and retrieves the most relevant
    text chunks based on a user query.

    Attributes:
        device (str): The device (either "cpu" or "cuda") used for computation.
        embeddings_csv (str): Path to the CSV file containing pre-computed embeddings.
        embeddings_df (pandas.DataFrame): DataFrame containing the embeddings.
        embedding_model (SentenceTransformer): A model used to encode the query into embeddings.
    """

    def __init__(self, embeddings_csv: str = "embeddings.csv"):
        """
        Initializes the Semantic_search class by loading the embeddings CSV and preparing
        the embedding model for semantic search.

        Args:
            embeddings_csv (str): Path to the CSV file containing pre-computed embeddings (default is "embeddings.csv").
        """
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Set device based on availability
        self.embeddings_csv = embeddings_csv  # Path to embeddings CSV
        self.embeddings_df = pd.read_csv(
            self.embeddings_csv
        )  # Load embeddings into DataFrame

        # Load pre-trained SentenceTransformer model
        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self.device
        )

    def _process_embeddings(self):
        """
        Processes the embeddings by converting the string representations of the embeddings
        into actual numpy arrays.

        This method is applied to the 'embedding' column of the embeddings DataFrame.
        """
        self.embeddings_df["embedding"] = self.embeddings_df["embedding"].apply(
            lambda x: np.fromstring(
                x.strip("[]"), sep=" "
            )  # Convert string to numpy array
        )

    def _get_pages_and_chunks_dict(self):
        """
        Converts the embeddings DataFrame into a list of dictionaries, where each dictionary
        contains the page number and the associated sentence chunk.

        Returns:
            list: A list of dictionaries containing page number and sentence chunk information.
        """
        pages_and_chunks = self.embeddings_df.to_dict(orient="records")
        return pages_and_chunks

    def _convert_embeddings_to_tensor(self):
        """
        Converts the embeddings DataFrame into a tensor format that can be used for computation
        with PyTorch.

        Returns:
            torch.Tensor: A tensor containing the embeddings.
        """
        return torch.tensor(
            np.array(self.embeddings_df["embedding"].tolist()), dtype=torch.float32
        ).to(self.device)

    def _retrieve_relevant_resources(
        self, query: str, embeddings: torch.tensor, n_resources_to_return: int = 5
    ):
        """
        Retrieves the most relevant resources based on the similarity between the query and the
        pre-computed embeddings using dot product similarity.

        Args:
            query (str): The user's query to search for relevant information.
            embeddings (torch.Tensor): The tensor of pre-computed embeddings.
            n_resources_to_return (int): The number of relevant resources to return (default is 5).

        Returns:
            tuple: A tuple containing the scores and indices of the top n most relevant resources.
        """
        query_embedding = self.embedding_model.encode(
            query, convert_to_tensor=True
        )  # Encode the query
        dot_scores = util.dot_score(query_embedding, embeddings)[
            0
        ]  # Calculate dot product scores
        scores, indices = torch.topk(
            input=dot_scores, k=n_resources_to_return
        )  # Get top results
        return scores, indices

    def _get_top_results(
        self,
        query: str,
        embeddings: torch.tensor,
        pages_and_chunks: list[dict],
        n_resources_to_return: int = 5,
    ):
        """
        Retrieves the top n relevant sentence chunks based on the query and pre-computed embeddings.

        Args:
            query (str): The user's query to search for relevant information.
            embeddings (torch.Tensor): The tensor of pre-computed embeddings.
            pages_and_chunks (list): The list of dictionaries containing page and sentence chunk information.
            n_resources_to_return (int): The number of relevant resources to return (default is 5).

        Returns:
            list: A list of the top n relevant sentence chunks.
        """
        relevant_chunks = []  # List to store relevant chunks
        scores, indices = self._retrieve_relevant_resources(
            query=query,
            embeddings=embeddings,
            n_resources_to_return=n_resources_to_return,
        )

        # Retrieve the relevant sentence chunks based on the indices
        for index in indices:
            sentence_chunk = pages_and_chunks[index]["sentence_chunk"]
            relevant_chunks.append(sentence_chunk)

        return relevant_chunks

    def run(self, query: str):
        """
        Runs the entire semantic search pipeline: processes the embeddings, retrieves relevant
        sentence chunks, and returns them based on the user's query.

        Args:
            query (str): The user's query to search for relevant information.

        Returns:
            list: A list of relevant sentence chunks based on the user's query.
        """
        self._process_embeddings()  # Process the embeddings to convert string to numpy arrays
        pages_and_chunks = (
            self._get_pages_and_chunks_dict()
        )  # Convert embeddings DataFrame to list of dictionaries
        embeddings_tensor = (
            self._convert_embeddings_to_tensor()
        )  # Convert embeddings to tensor
        relevant_chunks = self._get_top_results(
            query=query,
            embeddings=embeddings_tensor,
            pages_and_chunks=pages_and_chunks,
        )  # Retrieve the top relevant sentence chunks

        return relevant_chunks  # Return the relevant chunks
