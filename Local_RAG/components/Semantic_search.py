import numpy as np
import pandas as pd
import torch
from sentence_transformers import util, SentenceTransformer


class Semantic_Search:
    def __init__(self, embeddings_csv: str = "embeddings.csv"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_csv = embeddings_csv
        self.embeddings_df = pd.read_csv(self.embeddings_csv)

        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-mpnet-base-v2", device=self.device
        )

    def _process_embeddings(self):
        self.embeddings_df["embedding"] = self.embeddings_df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ")
        )

    def _get_pages_and_chunks_dict(self):
        pages_and_chucks = self.embeddings_df.to_dict(orient="records")

        return pages_and_chucks

    def _convert_embeddings_to_tensor(self):

        return torch.tensor(
            np.array(self.embeddings_df["embedding"].tolist()),
            dtype=torch.float32,
        ).to(self.device)

    def _retrieve_relevant_resources(
        self, query: str, embeddings: torch.tensor, n_resources_to_return: int = 5
    ):

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

        dot_scores = util.dot_score(query_embedding, embeddings)[0]

        scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

        return scores, indices

    def _get_top_results(
        self,
        query: str,
        embeddings: torch.tensor,
        pages_and_chunks: list[dict],
        n_resources_to_return: int = 5,
    ):
        relevant_chunks = []

        scores, indices = self._retrieve_relevant_resources(
            query=query,
            embeddings=embeddings,
            n_resources_to_return=n_resources_to_return,
        )

        for index in indices:
            sentence_chunk = pages_and_chunks[index]["sentence_chunk"]
            relevant_chunks.append(sentence_chunk)

        return relevant_chunks

    def run(self, query: str):
        self._process_embeddings()
        pages_and_chunks = self._get_pages_and_chunks_dict()
        embeddings_tensor = self._convert_embeddings_to_tensor()
        releveant_chuks = self._get_top_results(
            query=query,
            embeddings=embeddings_tensor,
            pages_and_chunks=pages_and_chunks,
        )

        return releveant_chuks


if __name__ == "__main__":
    semantic_search = Semantic_Search()
    releveant_chuks = semantic_search.run(query="breast feed")
