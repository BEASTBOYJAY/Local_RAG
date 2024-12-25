from Semantic_search import Semantic_search


class Create_prompt:
    """
    A class for creating a prompt based on a user query and context extracted through semantic search.

    This class handles the extraction of relevant context based on a query, formats the context into a
    prompt, and then generates a well-structured query-response template.

    Attributes:
        semantic_search (Semantic_search): An instance of the Semantic_search class for retrieving relevant context.
        base_prompt (str): A predefined template to structure the prompt with examples and a query.
    """

    def __init__(self):
        """
        Initializes the Create_prompt class with an instance of Semantic_search
        and a base prompt template for generating responses.
        """
        self.semantic_search = (
            Semantic_search()
        )  # Initialize the Semantic_search instance
        self.base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}\n
User query: {query}
Answer:"""

    def _get_releveant_chunks(self, query: str):
        """
        Retrieves relevant context chunks based on the user query using semantic search.

        Args:
            query (str): The user's question or query for which relevant context is needed.

        Returns:
            list: A list of relevant context chunks based on the query.
        """
        relevant_chunks = self.semantic_search.run(
            query=query
        )  # Run semantic search to find relevant context
        return relevant_chunks

    def _join_chunks(self, relevant_chunks: list):
        """
        Joins the list of relevant context chunks into a single string, formatted for the prompt.

        Args:
            relevant_chunks (list): A list of relevant context chunks.

        Returns:
            str: A string of context items formatted to be used in the prompt.
        """
        context = "- " + "\n- ".join(
            item for item in relevant_chunks
        )  # Join chunks with list item format
        return context

    def run(self, query: str):
        """
        Creates the full prompt by combining the base prompt template, relevant context, and user query.

        Args:
            query (str): The user's question or query.

        Returns:
            str: The complete prompt ready for input into a model or query system.
        """
        relevant_chunks = self._get_releveant_chunks(
            query=query
        )  # Get relevant context for the query
        context = self._join_chunks(relevant_chunks)  # Format the context into a string
        prompt = self.base_prompt.format(
            context=context, query=query
        )  # Format the base prompt with context and query
        return prompt
