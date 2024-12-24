from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from PDF_Processing import PDF_Processor
import pandas as pd


class Save_Embeddings:
    def __init__(self, embedding_model="all-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pdf_processor = PDF_Processor("human-nutrition-text.pdf")
        self.pages_and_chunks = self.pdf_processor.run()

        self.embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model, device=self.device
        )

    def _generate_embeddings(self):
        for item in tqdm(self.pages_and_chunks, desc="generating embeddings"):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])

    def _save_embeddings(self):
        data_frame = pd.DataFrame(self.pages_and_chunks)
        data_frame.to_csv("embeddings.csv", index=False)

    def run(self):
        self._generate_embeddings()
        self._save_embeddings()


if __name__ == "__main__":
    save_embeddings = Save_Embeddings()
    save_embeddings.run()
