from Embeddings import Save_Embeddings
from Prompter import Create_prompt
from LLM import LLM_Model
import os


class Local_RAG:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        if not os.path.exists("embeddings.csv"):
            self.save_embeddings = Save_Embeddings(pdf_path=self.pdf_path)
            self.save_embeddings.run()

        self.create_prompt = Create_prompt()
        self.llm_model = LLM_Model()

    def run(self, query):

        base_prompt = self.create_prompt.run(query=query)
        response = self.llm_model.run(base_prompt=base_prompt)
        return response


if __name__ == "__main__":
    local_rag = Local_RAG(pdf_path="human-nutrition-text.pdf")
    response = local_rag.run(query="what is the purpose of the book?")
    print(response)
