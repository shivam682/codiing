def find_similar_crs(state):
    jira_text = state["jira_text"]
    store = QdrantStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model
    )
    initial_results = store.similarity_search(jira_text, k=10)

    def hybrid_score(cr):
        cr_text = cr.page_content
        sem_score = cosine_similarity(
            [get_embedding(jira_text)],
            [get_embedding(cr_text)]
        )[0][0]
        fuzzy = fuzz.partial_ratio(jira_text, cr_text) / 100.0
        return 0.7 * sem_score + 0.3 * fuzzy

    reranked = sorted(initial_results, key=hybrid_score, reverse=True)[:5]
    return {"similar_crs": reranked, "jira_text": jira_text}
