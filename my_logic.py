import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import faiss
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel





# Load CSV

# Load FAISS index and metadata
index = faiss.read_index("iau_reviews_index.faiss")
with open("iau_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)



model = SentenceTransformer("HooshvareLab/bert-fa-zwnj-base")
# Load reviews CSV


# Load Persian tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
model = AutoModel.from_pretrained("HooshvareLab/bert-fa-zwnj-base").eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load FAISS index and metadata
index = faiss.read_index("iau_reviews_index.faiss")
with open("iau_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

def encode_texts(texts, batch_size=16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128).to(device)
            model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = sentence_embeddings.cpu().numpy()
            embeddings.append(sentence_embeddings)
    return np.vstack(embeddings)

def search_reviews(query, top_k=5):
    
    keywords = query.strip().split()

    candidate_rows = [
        r for r in metadata
        if any(kw in r["professor"] or kw in r["course"] for kw in keywords)
    ]

    if not candidate_rows:
        return []

    texts = [r["course"] + " " + r["professor"] + " " + r["comment"] for r in candidate_rows]
    vectors = encode_texts(texts)  
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    query_vec = encode_texts([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    
    local_index = faiss.IndexFlatIP(vectors.shape[1])
    local_index.add(vectors)

    D, I = local_index.search(query_vec, min(top_k, len(candidate_rows)))

    return [candidate_rows[i] for i in I[0]]


def filter_relevant(results, query):
    query = query.replace("؟", "").strip()
    query_tokens = set(query.split())

    def is_strict_match(row):
        # Normalize and tokenize professor and course
        prof_tokens = set(str(row["professor"]).strip().split())
        course_tokens = set(str(row["course"]).strip().split())

        # Match only if full token overlap exists (not substrings)
        match_prof = prof_tokens & query_tokens
        match_course = course_tokens & query_tokens

        return bool(match_prof or match_course)

    # Return all matching results
    return [r for r in results if is_strict_match(r)]





# ---- Fuzzy similarity score ----
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ---- Enhanced keyword fallback ----
def keyword_match_reviews(query, metadata):
    query = query.strip().replace("؟", "")
    keywords = set(query.split())

    results = []
    for row in metadata:
        prof = str(row["professor"])
        course = str(row["course"])
        for k in keywords:
            if k in prof or k in course or similar(k, prof) > 0.7 or similar(k, course) > 0.7:
                results.append(row)
                break
    return results

# ---- Sort by relevance ----
def relevance_score(row, query):
    score = 0
    if row["professor"] in query:
        score += 2
    if row["course"] in query:
        score += 2
    if row["professor"].split()[0] in query:
        score += 1
    if row["course"].split()[0] in query:
        score += 1
    return score

# ---- Strict context builder (best prof+course only) ----
def build_strict_context(reviews, user_question):
    prof_match_scores = defaultdict(int)
    course_match_scores = defaultdict(int)

    for r in reviews:
        prof_sim = similar(user_question, r["professor"])
        course_sim = similar(user_question, r["course"])
        if prof_sim > 0.6:
            prof_match_scores[r["professor"]] += prof_sim
        if course_sim > 0.6:
            course_match_scores[r["course"]] += course_sim

    best_prof = max(prof_match_scores, key=prof_match_scores.get, default="")
    best_course = max(course_match_scores, key=course_match_scores.get, default="")

    if best_prof and best_course:
        filtered = [
            r for r in reviews
            if similar(best_prof, r["professor"]) > 0.85 and similar(best_course, r["course"]) > 0.85
        ]
    elif best_course:
        filtered = [r for r in reviews if similar(best_course, r["course"]) > 0.85]
    elif best_prof:
        filtered = [r for r in reviews if similar(best_prof, r["professor"]) > 0.85]
    else:
        filtered = reviews


    result = f"👨‍🏫 استاد: {best_prof or '[نامشخص]'} — 📚 درس: {best_course or '[نامشخص]'}\n💬 نظرات:\n"
    for i, r in enumerate(filtered, 1):
        result += f"{i}. {r['comment'].strip()}\n🔗 لینک: {r['link']}\n\n"
    return result

# ---- Truncation helper ----
def truncate_reviews_to_fit(reviews, max_chars=127000):
    total = 0
    final = []
    for r in reviews:
        size = len(r["comment"])
        if total + size > max_chars:
            break
        final.append(r)
        total += size
    return final

# ---- Main answer function ----
def answer_question(user_question, gemini_model):

    print(f"\n🧠 Starting debug for question: {user_question}")

    retrieved = search_reviews(user_question, top_k=100)
    print(f"🔍 FAISS returned {len(retrieved)} raw rows")

    retrieved = filter_relevant(retrieved, user_question)
    print(f"✅ After filter_relevant(): {len(retrieved)} rows")

    keyword_hits = keyword_match_reviews(user_question, metadata)
    print(f"🔠 Keyword hits found: {len(keyword_hits)}")

    existing_links = set(r["link"] for r in retrieved)
    added = 0
    for r in keyword_hits:
        if r["link"] not in existing_links:
            retrieved.append(r)
            added += 1
    print(f"➕ Added {added} unique fallback keyword rows")
    print(f"📊 Total before truncation: {len(retrieved)}")

    if not retrieved:
        return "❌ هیچ تجربه‌ای در مورد سوال شما در داده‌های کانال یافت نشد."

    retrieved.sort(key=lambda r: relevance_score(r, user_question), reverse=True)
    retrieved = truncate_reviews_to_fit(retrieved)
    print(f"✂️ After truncation: {len(retrieved)} rows")

    context = build_strict_context(retrieved, user_question)
    print("📝 Sample context sent to GPT:\n", context[:100000], "\n...")

    prompt = f"""شما یک دستیار هوشمند انتخاب واحد هستید که فقط و فقط بر اساس نظرات واقعی دانشجویان از کانال @IAUCourseExp پاسخ می‌دهید. کار شما کمک به دانشجویان برای انتخاب استاد و درس، بر اساس تجربیات ثبت‌شده در این کانال است.

❗ قوانین مهم:
- فقط از داده‌های همین نظرات استفاده کن. هیچ اطلاعات اضافی، حدسی یا اینترنتی استفاده نکن.
- اگر هیچ نظری درباره سؤال وجود ندارد، فقط بگو: «هیچ تجربه‌ای دربارهٔ این مورد در کانال ثبت نشده است.»
- سوالات دانشجو می‌توانند از انواع مختلف باشند:
  • بررسی یک استاد خاص
  • مقایسه چند استاد برای یک درس
  • معرفی بهترین یا بدترین استادهای یک درس
  • تحلیل نظر کلی دانشجویان درمورد یک درس خاص
  بنابراین آماده باش که با توجه به داده‌ها به هر نوع سوال، دقیق و قابل اعتماد پاسخ بدهی.
- همه‌ی نظرات مربوط به سوال را بررسی کن (نه فقط یکی یا دو تا) و به‌صورت فهرست‌وار یا خلاصه‌شده تحلیلشان کن.
- برای هر نظر، لینک تلگرام مربوطه را نیز حتماً ذکر کن.
- در پایان پاسخ، نتیجه‌گیری نهایی خود را بنویس: آیا این استاد برای این درس توصیه می‌شود یا نه — فقط بر اساس همین نظرات.
- در انتها حتماً بنویس:
📊 این پاسخ بر اساس بررسی {len(retrieved)} نظر دانشجویی نوشته شده است.

🔎 سوال دانشجو:
{user_question}

📄 نظرات دانشجویان (برگرفته از کانال تجربیات انتخاب واحد):
{context}

📘 پاسخ نهایی:
"""


    # NEW (Gemini)

    response = gemini_model.generate_content(prompt)
    return response.text
