from Semantic_search import Semantic_search


class Create_prompt:
    def __init__(self):
        self.semantic_search = Semantic_search()
        self.base_promt = """Based on the following context items, please answer the query.
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
        releveant_chunks = self.semantic_search.run(query=query)

        return releveant_chunks

    def _join_chunks(self, relevant_chunks: list):
        context = "- " + "\n- ".join(item for item in relevant_chunks)
        return context

    def run(self, query: str):
        relevant_chunks = self._get_releveant_chunks(query=query)
        context = self._join_chunks(relevant_chunks)
        prompt = self.base_promt.format(context=context, query=query)
        return prompt


if __name__ == "__main__":
    create_prompt = Create_prompt()
    print(
        create_prompt.run(
            query="What are the macronutrients, and what roles do they play in the human body?"
        )
    )
