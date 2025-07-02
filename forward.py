# Fast Log Matching using MinHash + LSH (Plagiarism Detection Style)

import os
import argparse
import pickle
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from tqdm import tqdm

# --- CONFIG ---
DUMP_DIR = "/mnt/data/new_jira_dump"
INDEX_FILE = "log_minhash_index.pkl"

# --- PARAMETERS ---
NUM_PERM = 128     # Number of permutations for MinHash
LSH_THRESHOLD = 0.3  # Lower threshold to retrieve more candidates
STRIDE = 5          # Overlap between shingles

# --- FUNCTIONS ---
def file_to_shingles(filepath, shingle_size, stride=STRIDE):
    with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()
    shingles = []
    for i in range(0, len(lines) - shingle_size + 1, stride):
        shingle = "".join(lines[i:i + shingle_size]).strip()
        if shingle:
            shingles.append((shingle, filepath, i))  # Include source info
    return shingles

def create_minhash(text, num_perm=NUM_PERM):
    mh = MinHash(num_perm=num_perm)
    for word in text.split():
        mh.update(word.encode("utf-8"))
    return mh

def index_logs(dump_dir, shingle_size):
    shingles_map = {}
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)

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

    shingle_size = len(snippet.splitlines())
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
    parser.add_argument("--shingle", type=int, default=20, help="Shingle size (lines per block)")
    args = parser.parse_args()

    if args.index:
        index_logs(DUMP_DIR, shingle_size=args.shingle)
    elif args.snippet:
        find_best_match(args.snippet)
    else:
        print("Use --index to index logs or --snippet to search with a snippet.")

