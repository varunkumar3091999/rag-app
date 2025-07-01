import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

model = SentenceTransformer("all-MiniLM-L6-v2")
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

text_file_path = "context.txt"
embeddings_file_path = "saved_embeddings.pkl"

def chunk_text(text, chunk_size=10):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

if os.path.exists(embeddings_file_path):
    with open(embeddings_file_path, "rb") as f:
        chunk_embeddings, chunks = pickle.load(f)
    print("✅ Embeddings loaded from file.")
else:
    with open(text_file_path, "r") as f:
        text = f.read()

    chunks = chunk_text(text)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

    with open(embeddings_file_path, "wb") as f:
        pickle.dump((chunk_embeddings, chunks), f)
    print("✅ Embeddings computed and saved.")

while True:
    question = input("\n❓ Your question (type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top_idx = int(np.argmax(scores))
    context = chunks[top_idx]
    
    prompt = f"context: {context}\n\nQuestion: {question}\\Answer:"
    
    answer = gen_pipeline(prompt, max_length=200)[0]['generated_text']

    print("\n🧠 Answer:")
    print(answer)
