from adaptive_rag import run_adaptive_rag_workflow

print("test_run.py loaded")

if __name__ == "__main__":
    print("main block running")

    query = "How is retrieval augmented generation used in academic research systems?"
    result = run_adaptive_rag_workflow(query)

    print("\n" + "=" * 80)
    print("ADAPTIVE RAG WORKFLOW RESULT")
    print("=" * 80)

    print("\nUSER QUERY:")
    print(result.get("user_query"))

    print("\nVALIDATED QUERY:")
    print(result.get("validated_query"))

    print("\nREFINED QUERY:")
    print(result.get("refined_query") or "No refinement applied")

    print("\nRETRIEVAL QUERIES:")
    for q in result.get("retrieval_queries", []):
        print("-", q)

    print("\nRETRIEVED DOCUMENTS:")
    retrieved_docs = result.get("retrieved_documents", [])
    if not retrieved_docs:
        print("No documents retrieved.")
    else:
        for idx, doc in enumerate(retrieved_docs, start=1):
            print(f"{idx}. {doc.get('title', 'Untitled')}")
            if doc.get("authors"):
                print("   Authors:", ", ".join(doc.get("authors", [])))
            if doc.get("published"):
                print("   Published:", doc.get("published"))
            if doc.get("link"):
                print("   Link:", doc.get("link"))

    print("\nRELEVANT DOCUMENTS:")
    relevant_docs = result.get("relevant_documents", [])
    if not relevant_docs:
        print("No relevant documents found.")
    else:
        for idx, doc in enumerate(relevant_docs, start=1):
            print(f"{idx}. {doc.get('title', 'Untitled')}")
            print("   Label:", doc.get("relevance_label"))
            print("   Score:", doc.get("relevance_score"))
            matched_terms = doc.get("matched_terms", [])
            if matched_terms:
                print("   Matched terms:", ", ".join(matched_terms))
            if doc.get("link"):
                print("   Link:", doc.get("link"))

    print("\nDISCARDED DOCUMENTS:")
    discarded_docs = result.get("discarded_documents", [])
    if not discarded_docs:
        print("No discarded documents.")
    else:
        for idx, doc in enumerate(discarded_docs, start=1):
            print(f"{idx}. {doc.get('title', 'Untitled')}")
            print("   Label:", doc.get("relevance_label"))
            print("   Score:", doc.get("relevance_score"))

    print("\nFINAL ANSWER:")
    print(result.get("final_answer"))

    print("\nCONFIDENCE:")
    print(result.get("confidence_note"))

    print("\nMEMORY NOTES:")
    for note in result.get("memory_notes", []):
        print("-", note)

    print("\n" + "=" * 80)