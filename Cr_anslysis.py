def generate_final_json_score(analysis_dir="cr_analysis_reports"):
    all_analyses = []
    for fname in os.listdir(analysis_dir):
        if not fname.endswith(".txt"):
            continue
        cr_number = fname.replace(".txt", "")
        with open(os.path.join(analysis_dir, fname), "r", encoding="utf-8") as f:
            content = f.read()
            all_analyses.append((cr_number, content))

    combined_prompt = "\n\n".join([
        f"CR Number: {cr}\nAnalysis:\n{text}" for cr, text in all_analyses
    ])

    system_prompt = (
        "You are an expert crash analyst.\n"
        "Based on the analyses of different CRs vs a new JIRA crash, give similarity scores "
        "between 0 and 10 along with a justification.\n\n"
        "Output JSON format:\n"
        "[\n  {\n    \"cr_number\": \"...\",\n    \"score\": float,\n    \"justification\": \"...\"\n  },\n  ...\n]"
    )

    response = llm.invoke(
        input=combined_prompt,
        system_message=system_prompt
    )

    print("\n=== Final Similarity Scores ===\n")
    print(response)
    return response
