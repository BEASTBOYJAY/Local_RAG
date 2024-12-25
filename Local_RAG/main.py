from components.RAG import Local_RAG

pdf_path = r"your/pdf/path"
local_rag = Local_RAG(pdf_path=pdf_path)
query = "What is the purpose of the paper?"
local_rag.run(query=query)
