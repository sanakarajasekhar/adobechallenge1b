import os
import json
import time
import re
from typing import List, Dict

import fitz
from sentence_transformers import SentenceTransformer, util
import faiss

CHUNK_SIZE = 512
TOP_K = 10

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_by_page(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    sections = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if len(text.strip().split()) < 20:
            continue
        sections.append({
            "document": os.path.basename(pdf_path),
            "page": page_num + 1,
            "text": text.strip()
        })
    return sections

def chunk_sections(sections: List[Dict]) -> List[Dict]:
    chunked = []
    for sec in sections:
        words = sec["text"].split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk_text = " ".join(words[i:i + CHUNK_SIZE])
            chunked.append({
                "document": sec["document"],
                "page": sec["page"],
                "text": chunk_text
            })
    return chunked

def extract_clean_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    candidates = re.split(r"\n|\.\s|: |\u2022| - ", text)
    for cand in candidates:
        cand = cand.strip(". :-\n")
        if 25 <= len(cand) <= 100 and not cand.isdigit():
            return cand
    return text[:90].strip()

def build_index(chunks: List[Dict]):
    texts = [x["text"] for x in chunks]
    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return index, embeddings, chunks

def search_relevant_sections(query: str, index, embeddings, chunks):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    matches = util.semantic_search(query_embedding, embeddings, top_k=TOP_K)[0]
    results = []
    for rank, match in enumerate(matches, 1):
        idx = match["corpus_id"]
        score = match["score"]
        c = chunks[idx]

        title = extract_clean_title(c["text"])

        results.append({
            "document": c["document"],
            "page": c["page"],
            "section_title": title,
            "importance_rank": rank,
            "score": round(float(score), 4),
            "refined_text": c["text"][:400]
        })
    return results

def process_documents(input_dir, output_dir) -> Dict:
    input_json_path = os.path.join(output_dir, "challenge1b_input.json")
    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    docs = [doc["filename"] for doc in input_data["documents"]]

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    all_sections = []

    for pdf in docs:
        pdf_path = os.path.join(input_dir, pdf)
        if not os.path.exists(pdf_path):
            print(f"⚠️ File not found: {pdf_path}")
            continue
        sections = extract_text_by_page(pdf_path)
        all_sections.extend(sections)

    chunks = chunk_sections(all_sections)[:300]
    index, embeddings, chunks_used = build_index(chunks)

    query = f"For a {persona}, tasked to {job}, what parts of the document matter most?"
    matches = search_relevant_sections(query, index, embeddings, chunks_used)

    extracted_sections = [
        {
            "document": m["document"],
            "section_title": m["section_title"],
            "importance_rank": m["importance_rank"],
            "page_number": m["page"]
        } for m in matches
    ]

    subsection_analysis = [
        {
            "document": m["document"],
            "refined_text": m["refined_text"],
            "page_number": m["page"]
        } for m in matches
    ]

    return {
        "metadata": {
            "input_documents": docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python adobe_hackathon_ai.py <input_dir> <output_dir>")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    result = process_documents(input_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "challenge1b_output.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Process complete. challenge1b_output.json generated.")
