from Embeddings import Save_Embeddings
from Prompter import Create_prompt
from LLM import LLM_Model
import os


class Local_RAG:
    """
    A class for implementing a Local Retrieval-Augmented Generation (RAG) model. This model retrieves context
    from a PDF, creates a prompt based on the query, and then uses a language model to generate a response.

    Attributes:
        pdf_path (str): The path to the PDF document to be processed.
        save_embeddings (Save_Embeddings): An instance of Save_Embeddings to save embeddings from the PDF.
        create_prompt (Create_prompt): An instance of Create_prompt to generate a prompt based on the query.
        llm_model (LLM_Model): An instance of LLM_Model to run the language model for generating responses.
    """

    def __init__(self, pdf_path):
        """
        Initializes the Local_RAG class. If the embeddings CSV file does not exist,
        embeddings are generated and saved from the given PDF document.

        Args:
            pdf_path (str): The path to the PDF file to be processed.
        """
        self.pdf_path = pdf_path  # Path to the PDF file

        # Check if the embeddings CSV file already exists. If not, generate and save embeddings.
        if not os.path.exists("embeddings.csv"):
            self.save_embeddings = Save_Embeddings(
                pdf_path=self.pdf_path
            )  # Initialize Save_Embeddings
            self.save_embeddings.run()  # Run the embedding saving process

        self.create_prompt = (
            Create_prompt()
        )  # Initialize Create_prompt for prompt generation
        self.llm_model = (
            LLM_Model()
        )  # Initialize LLM_Model for language model response generation

    def run(self, query):
        """
        Runs the Local_RAG process: generates a prompt based on the query, retrieves the relevant response
        from the language model, and returns the response.

        Args:
            query (str): The user's query to be answered.

        Returns:
            str: The generated response from the language model based on the context and query.
        """
        print("Creating Prompt....")
        base_prompt = self.create_prompt.run(
            query=query
        )  # Generate the base prompt using the query
        print("Generating Results....")
        response = self.llm_model.run(
            base_prompt=base_prompt
        )  # Generate a response from the language model
        return response  # Return the generated response
