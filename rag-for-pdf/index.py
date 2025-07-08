import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from read_pdf import extract_text_from_pdf, clean_text, split_into_chunks, remove_header_and_footer
import torch

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
# tokenizer = AutoTokenizer.from_pretrained(llm_name)



llm_name = "gpt2"
llm_tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
llm_model = GPT2LMHeadModel.from_pretrained(llm_name)
llm_model.eval()

if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm_model = llm_model.to(device)

# text_file_path = "context.txt"
embeddings_file_path = "saved_embeddings.pkl"
pdf_path = 'sample.pdf'



# def chunk_text(text, chunk_size=100):
#     words = text.split()
#     return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

if os.path.exists(embeddings_file_path):
    with open(embeddings_file_path, "rb") as f:
        loaded_embeddings = pickle.load(f)
    embeddings_array, chunks = loaded_embeddings

    print(embeddings_array, "Loaded embeddings")
    print(embeddings_array.shape)
    print(len(loaded_embeddings), 'length')
    print("✅ Embeddings loaded from file.")
else:
    full_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(full_text)
    headers_removed = remove_header_and_footer(cleaned_text, "MBA – Organizational Behavior – RMB1C - 1st Sem By, Shelpa (A24)")
    chunks = split_into_chunks(headers_removed)
    chunk_embeddings = embed_model.encode(chunks, convert_to_tensor=True)


    print(len(chunks), "chunks length")
    print(chunk_embeddings.shape)
    
    # print(chunk_embeddings)
    with open(embeddings_file_path, "wb") as f:
        pickle.dump((chunk_embeddings, chunks), f)
    print("✅ Embeddings computed and saved.")



# while True:
#     question = input("\n❓ Your question (type 'exit' to quit): ")
#     if question.lower() == "exit":
#         break

#     question_embedding = embed_model.encode(question, convert_to_tensor=True)
#     scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
#     # top_idx = int(np.argmax(scores))
#     top_k = min(10,len(chunks))
#     top_results = torch.topk(scores,k=top_k)
#     selected_chunks = [chunks[idx] for idx in top_results[1]]
#     combined_context = "\n\n".join(selected_chunks)
#     # context = chunks[top_idx]
        
#     max_context_tokens = 800
#     context_tokens = llm_tokenizer.encode(combined_context, add_special_tokens=False)
    
#     if len(context_tokens) > max_context_tokens:
#         context_tokens = context_tokens[:max_context_tokens]

#     context = llm_tokenizer.decode(context_tokens, skip_special_tokens=True)

    
#     prompt = f"Use the context below to answer the question clearly and concisely.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"


    
    
#     inputs = llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1600).to(device)
#     # input_ids = inputs["input_ids"]
#     # attention_mask = inputs["attention_mask"]
    
    
#     total_tokens = len(llm_tokenizer.encode(prompt, add_special_tokens=False))
#     if total_tokens > 1024:
#         print(f"⚠️ Input too long ({total_tokens} tokens), truncating.")
        
        
#     output = llm_model.generate(
#       **inputs,
#     #   attention_mask=attention_mask,
#       max_new_tokens=150,
#       do_sample=True,
#     #   temprature=0.7,
#       top_p=0.9,
#       pad_token_id=llm_tokenizer.eos_token_id
#     )
#     # decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     # print(decoded_output, "dec")
    
#     # if "Answer:" in decoded_output:
#     #   final_answer = decoded_output.split("Answer:")[-1].strip()
#     # else:
#     #   final_answer = decoded_output.strip()
    
#     generated_text = llm_tokenizer.decode(output[0],skip_special_tokens=True)
    
#     answer = generated_text[len(prompt):].strip()
#     # output = llm_model(inputs)

    
#     # answer = gen_pipeline(prompt, max_new_tokens=500)[0]['generated_text']
#     print(answer if answer else "[Could not generate a clear answer.]")
