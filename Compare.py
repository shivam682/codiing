# Fast Log Matching using MinHash + LSH (Plagiarism Detection Style) with Multiprocessing and Top-K Fuzzy Matching

import os
import argparse
import pickle
import time
import json
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- FUNCTIONS ---
def file_to_shingles(filepath, shingle_size, stride):
    with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()
    shingles = []
    for i in range(0, len(lines) - shingle_size + 1, stride):
        shingle = "".join(lines[i:i + shingle_size]).strip()
        if shingle:
            shingles.append((shingle, filepath, i))
    return shingles

def create_minhash(text, num_perm):
    mh = MinHash(num_perm=num_perm)
    for word in text.split():
        mh.update(word.encode("utf-8"))
    return mh

def process_file(args):
    filepath, shingle_size, stride, num_perm = args
    shingles = file_to_shingles(filepath, shingle_size, stride)
    results = []
    for idx, (text, src_file, line_num) in enumerate(shingles):
        mh = create_minhash(text, num_perm)
        key = f"{src_file}:{line_num}:{idx}"
        results.append((key, mh, text))
    return results

def index_logs(dump_dir, index_file, shingle_size, stride, threshold, num_perm):
    start_time = time.time()
    shingles_map = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    file_list = [os.path.join(root, file)
                 for root, _, files in os.walk(dump_dir)
                 for file in files]

    args = [(f, shingle_size, stride, num_perm) for f in file_list]
    with Pool(processes=cpu_count()) as pool:
        for file_result in tqdm(pool.imap_unordered(process_file, args), total=len(args), desc="Indexing files"):
            for key, mh, text in file_result:
                lsh.insert(key, mh)
                shingles_map[key] = text

    with open(index_file, "wb") as f:
        pickle.dump((lsh, shingles_map), f)
    print("Indexing complete.")
    print(f"Total indexing time: {time.time() - start_time:.2f} seconds")

def fuzzy_score(args):
    snippet, key, text = args
    score = fuzz.partial_ratio(snippet, text)
    return (key, score, text)

def find_best_match_for_snippet(snippet, lsh, shingles_map, num_perm, top_k):
    snippet_text = "\n".join(snippet).strip()
    snippet_mh = create_minhash(snippet_text, num_perm)
    candidates = lsh.query(snippet_mh)

    if not candidates:
        candidates = list(shingles_map.keys())

    scored_candidates = [(key, len(set(snippet_text.split()) & set(shingles_map[key].split()))) for key in candidates]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [key for key, _ in scored_candidates[:top_k]]

    args = [(snippet_text, key, shingles_map[key]) for key in top_candidates]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(fuzzy_score, args)

    best = max(results, key=lambda x: x[1])
    return {"snippet": snippet_text, "match": best[2], "score": best[1], "source": best[0]}

def find_best_matches(snippet_json_path, index_file, num_perm, top_k):
    start_time = time.time()
    with open(index_file, "rb") as f:
        lsh, shingles_map = pickle.load(f)

    with open(snippet_json_path, "r", encoding="utf-8") as f:
        all_snippets = json.load(f)  # List of list of lines

    args = [(snippet, lsh, shingles_map, num_perm, top_k) for snippet in all_snippets]
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(find_best_match_for_snippet, args)

    for result in results:
        print(f"\n---\nBest Match [Score: {result['score']}] from {result['source']}\nSnippet:\n{result['snippet']}\nMatch:\n{result['match']}\n")
    print(f"Total retrieval time: {time.time() - start_time:.2f} seconds")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Run indexing")
    parser.add_argument("--snippet", type=str, help="Path to single snippet file")
    parser.add_argument("--snippet_json", type=str, help="Path to snippet JSON (list of list of lines)")
    parser.add_argument("--dump_dir", type=str, default="/mnt/data/new_jira_dump", help="Directory with dump files")
    parser.add_argument("--index_file", type=str, default="log_minhash_index.pkl", help="Path to store/read MinHash index")
    parser.add_argument("--shingle", type=int, default=20, help="Shingle size (lines per block)")
    parser.add_argument("--stride", type=int, default=5, help="Stride or overlap between shingles")
    parser.add_argument("--top_k", type=int, default=10, help="Top K results for fuzzy scoring")
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations for MinHash")
    parser.add_argument("--threshold", type=float, default=0.5, help="LSH threshold (higher = stricter match)")
    args = parser.parse_args()

    if args.index:
        index_logs(args.dump_dir, args.index_file, args.shingle, args.stride, args.threshold, args.num_perm)
    elif args.snippet:
        find_best_match_for_snippet(args.snippet, args.index_file, args.num_perm, args.top_k)
    elif args.snippet_json:
        find_best_matches(args.snippet_json, args.index_file, args.num_perm, args.top_k)
    else:
        print("Use --index to index logs, --snippet for a single snippet file, or --snippet_json for multiple snippets.")


✅ Your code is now updated to support parallel processing for multiple snippets via a JSON input (list of list of lines).

New CLI Usage:

To index logs:

python script.py --index

To search for a single snippet (from a file):

python script.py --snippet path/to/snippet.txt

To search for multiple snippets (from a JSON file):

python script.py --snippet_json path/to/snippets.json


Each entry in the JSON should be a list of lines like:

[
  ["line 1 of snippet 1", "line 2 of snippet 1"],
  ["line 1 of snippet 2", "line 2 of snippet 2", "line 3 of snippet 2"]
]

Let me know if you want to output these results in a structured file like JSON or CSV!

              
