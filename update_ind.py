def hybrid_score(text1, text2):
    """Combine cosine and fuzzy score"""
    try:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        cos_sim = cosine_similarity([emb1], [emb2])[0][0]
    except:
        cos_sim = 0

    fuzz_sim = fuzz.partial_ratio(text1, text2) / 100.0
    return 0.6 * cos_sim + 0.4 * fuzz_sim  # adjust weights as needed

def find_similar_crs(state):
    jira_text = state["jira_text"]
    store = QdrantStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model
    )
    
    initial_results = store.similarity_search(jira_text, k=10)

    # Refine results using hybrid scoring
    scored = []
    for cr in initial_results:
        score = hybrid_score(jira_text, cr.page_content)
        scored.append((score, cr))
    
    top_3 = sorted(scored, key=lambda x: x[0], reverse=True)[:3]
    return {
        "similar_crs": [cr for _, cr in top_3],
        "jira_text": jira_text
    }

