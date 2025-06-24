def match_log_snippet_in_dump(snippet_lines, window_size=10):
    """
    snippet_lines: List[str] - lines from the old CR log group
    """
    snippet = "\n".join(snippet_lines).strip()
    if not snippet:
        return "Snippet is empty."

    best_match = ("", 0.0, "", 0)  # content, score, filename, line number
    snippet_embedding = get_embedding(snippet)

    for root, _, files in os.walk(NEW_JIRA_DUMP_PATH):
        for file in files:
            fpath = os.path.join(root, file)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    for i in range(len(lines) - window_size + 1):
                        block_lines = lines[i:i+window_size]
                        block = "\n".join(line.strip() for line in block_lines).strip()
                        if not block:
                            continue
                        block_embedding = get_embedding(block)
                        sem_score = cosine_similarity([snippet_embedding], [block_embedding])[0][0]
                        fuzzy = fuzzy_score(snippet, block)
                        score = max(sem_score, fuzzy)
                        if score > best_match[1]:
                            best_match = (block, score, file, i)
            except:
                continue

    return f"Best Match in {best_match[2]} (starting at line {best_match[3]}) [Score: {best_match[1]:.2f}]:\n{best_match[0]}"
