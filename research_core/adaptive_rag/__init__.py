from .workflow import run_adaptive_rag_workflow

# from adaptive_rag import run_adaptive_rag_workflow


# if __name__ == "__main__":
#     query = "What is adaptive RAG in research assistants?"
#     result = run_adaptive_rag_workflow(query)

#     print("USER QUERY:")
#     print(result.get("user_query"))

#     print("\nVALIDATED QUERY:")
#     print(result.get("validated_query"))

#     print("\nREFINED QUERY:")
#     print(result.get("refined_query"))

#     print("\nRETRIEVED DOCUMENTS:")
#     for doc in result.get("retrieved_documents", []):
#         print("-", doc.get("title"))

#     print("\nRELEVANT DOCUMENTS:")
#     for doc in result.get("relevant_documents", []):
#         print("-", doc.get("title"))

#     print("\nFINAL ANSWER:")
#     print(result.get("final_answer"))

#     print("\nCONFIDENCE:")
#     print(result.get("confidence_note"))

#     print("\nMEMORY NOTES:")
#     for note in result.get("memory_notes", []):
#         print("-", note)