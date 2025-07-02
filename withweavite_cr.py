# Fast Log Matching using MinHash + LSH (Plagiarism Detection Style) with Multiprocessing and Top-K Fuzzy Matching

import os
import argparse
import pickle
import time
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

def find_best_match(snippet_path, index_file, num_perm, top_k):
    start_time = time.time()
    with open(index_file, "rb") as f:
        lsh, shingles_map = pickle.load(f)

    with open(snippet_path, "r", encoding="utf-8") as f:
        snippet = f.read().strip()

    snippet_mh = create_minhash(snippet, num_perm)
    candidates = lsh.query(snippet_mh)

    if not candidates:
        print("No candidates found with LSH, fallback to full scan.")
        candidates = list(shingles_map.keys())

    scored_candidates = [(key, len(set(snippet.split()) & set(shingles_map[key].split()))) for key in candidates]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [key for key, _ in scored_candidates[:top_k]]

    args = [(snippet, key, shingles_map[key]) for key in top_candidates]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(fuzzy_score, args)

    best = max(results, key=lambda x: x[1])
    print(f"Best Match [Score: {best[1]}] from {best[0]}:\n{best[2]}")
    print(f"Retrieval time: {time.time() - start_time:.2f} seconds")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Run indexing")
    parser.add_argument("--snippet", type=str, help="Path to log snippet file")
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
        find_best_match(args.snippet, args.index_file, args.num_perm, args.top_k)
    else:
        print("Use --index to index logs or --snippet to search with a snippet.")
