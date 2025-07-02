# Fast Log Matching using MinHash + LSH (Plagiarism Detection Style)

import os
import argparse
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from tqdm import tqdm

# --- CONFIG ---
DUMP_DIR = "/mnt/data/new_jira_dump"
INDEX_FILE = "log_minhash_index.pkl"

# --- PARAMETERS ---
shingle_size = 5  # Number of lines per shingle
num_perm = 128    # Number of permutations for MinHash
lsh_threshold = 0.3  # Lower threshold to retrieve more candidates

import pickle

# --- FUNCTIONS ---
def file_to_shingles(filepath, shingle_size):
    with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()
    shingles = []
    for i in range(0, len(lines) - shingle_size + 1):
        shingle = "".join(lines[i:i + shingle_size]).strip()
        if shingle:
            shingles.append((shingle, filepath, i))  # Include source info
    return shingles

def create_minhash(text, num_perm=num_perm):
    mh = MinHash(num_perm=num_perm)
    for word in text.split():
        mh.update(word.encode("utf-8"))
    return mh

def index_logs(dump_dir):
    shingles_map = {}
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)

    for root, _, files in os.walk(dump_dir):
        for file in tqdm(files, desc="Indexing files"):
            fpath = os.path.join(root, file)
            shingles = file_to_shingles(fpath, shingle_size)
            for idx, (text, src_file, line_num) in enumerate(shingles):
                mh = create_minhash(text)
                key = f"{src_file}:{line_num}:{idx}"
                lsh.insert(key, mh)
                shingles_map[key] = text

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((lsh, shingles_map), f)
    print("Indexing complete.")

def find_best_match(snippet_path):
    with open(INDEX_FILE, "rb") as f:
        lsh, shingles_map = pickle.load(f)

    with open(snippet_path, "r", encoding="utf-8") as f:
        snippet = f.read().strip()

    snippet_mh = create_minhash(snippet)
    candidates = lsh.query(snippet_mh)

    if not candidates:
        print("No candidates found with LSH, fallback to full scan.")
        candidates = list(shingles_map.keys())

    best = ("", 0, "")
    for key in candidates:
        text = shingles_map[key]
        score = fuzz.partial_ratio(snippet, text)
        if score > best[1]:
            best = (text, score, key)

    print(f"Best Match [Score: {best[1]}] from {best[2]}:\n{best[0]}")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Run indexing")
    parser.add_argument("--snippet", type=str, help="Path to log snippet file")
    args = parser.parse_args()

    if args.index:
        index_logs(DUMP_DIR)
    elif args.snippet:
        find_best_match(args.snippet)
    else:
        print("Use --index to index logs or --snippet to search with a snippet.")


I've replaced your original CR agent pipeline with an optimized version using the MinHash + LSH (plagiarism-style) approach for fast and scalable log matching. You can now:

Run python script.py --index to index all logs from the dump directory.

Run python script.py --snippet path_to_snippet.txt to find the best matching log block for your snippet.


Let me know if you want to re-integrate this into your CR comparison workflow or add score-based JSON output!

