import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import wikipediaapi
from sentence_transformers import SentenceTransformer


class WikiRAG:
    def __init__(
        self,
        wiki_page,
        api_key_env_var="API_KEY",
        wiki_lang="Simple_RAG",
        model_name="all-MiniLM-L6-v2",
        groq_model="llama-3.3-70b-versatile",
    ):
        """Initializes the WikiRAG system."""
        load_dotenv()
        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError("API Key not found in environment variables.")

        self.client = Groq(api_key=self.api_key)

        self.wiki_wiki = wikipediaapi.Wikipedia(wiki_lang, "en")
        self.page_content = self._get_wiki_page_content(wiki_page)
        self.paragraphs = self.page_content.split("\n\n")

        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.doc_embeddings = self._embed_paragraphs(self.paragraphs)

        self.groq_model = groq_model

    def _get_wiki_page_content(self, page_title):
        """Fetches and returns the content of the specified Wikipedia page."""
        page = self.wiki_wiki.page(page_title)
        if not page.exists():
            raise ValueError(f"Page '{page_title}' does not exist on Wikipedia.")
        return page.text

    def _embed_paragraphs(self, paragraphs):
        """Encodes and normalizes the paragraph embeddings."""
        return self.model.encode(paragraphs, normalize_embeddings=True)

    def _embed_query(self, query):
        """Encodes and normalizes the query embedding."""
        return self.model.encode(query, normalize_embeddings=True)

    def _find_most_similar_paragraphs(self, query, top_k=3):
        """Finds the top-k most similar paragraphs for the given query."""
        query_embed = self._embed_query(query)
        similarity = np.dot(self.doc_embeddings, query_embed.T)
        top_k_indices = np.argsort(similarity, axis=0)[-top_k:][::-1].tolist()
        most_similar_docs = [self.paragraphs[idx] for idx in top_k_indices]
        return most_similar_docs

    def _build_prompt(self, context, query):
        """Builds the prompt to be sent to the LLM."""
        return f"""
            Answer the question based on the context below. If you cannot answer the question, reply "I don't know".

            Context: {context}

            Question: {query}
        """

    def generate_answer(self, query, top_k=3):
        """Generates an answer to the query using Groq's LLM."""
        most_similar_docs = self._find_most_similar_paragraphs(query, top_k=top_k)
        context = "\n\n".join(most_similar_docs)
        prompt = self._build_prompt(context, query)

        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model=self.groq_model,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    rag_system = WikiRAG(wiki_page="Ai")
    query = "hii"
    answer = rag_system.generate_answer(query)
    print(answer)
