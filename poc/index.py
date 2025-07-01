from sentence_transformers  import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-V2')

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_answer(context_chunks, question, max_tokens=200):
    context = "\n".join(context_chunks)
    prompt = f"Answer the question based on the context:\n\nContext: \n{context}: {question}"
    
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True)
    output = llm.generate(**inputs, max_new_tokens = max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_chunks(file, chunk_size=500):
    with open(file, 'r',encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    chunks = [" ".join(words[1:1 + chunk_size]) for i in range(0, len(words),chunk_size)]
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks)

def embed_question(question):
    return model.encode([question])[0]

def retrieve_top_chunks(question_embedding, chunk_embeddings, chunks, k=3):
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [chunks[i] for i in top_indices]


def simulate_answer(context_chunks, question):
    context = "\n---\n".join(context_chunks)
    return f"[SIMULATED ANSWER]\n\nQuestion: {question}\n\nContext:\n{context}"

if __name__ == "__main__":
    filepath = "context.txt"
    chunks = load_chunks(filepath)
    chunk_embeddings = embed_chunks(chunks)
    
    print("Text Loaded and embedded. Ask me anything about it")
    
    while True:
        question = input("Your question (type 'exit' to quit): ")
        if question.strip().lower() in ['exit']:
            break
        
        q_embeddings = embed_question(question)
        top_chunks = retrieve_top_chunks(q_embeddings, chunk_embeddings, chunks)
        answer = generate_answer(top_chunks, question)
    
        print("\n🧠 Answer:")
        print(answer)
        print("\n" + "=" * 80 + "\n")