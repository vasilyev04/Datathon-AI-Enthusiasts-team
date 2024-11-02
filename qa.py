from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
repo_id = 'nur-dev/roberta-large-kazqad'
model = AutoModelForQuestionAnswering.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Define the context and question
context = """
Алматы Қазақстанның ең ірі мегаполисі. Алматы – асқақ Тянь-Шань тауы жотасының көкжасыл бауырайынан, 
Іле Алатауының бөктерінде, Қазақстан Республикасының оңтүстік-шығысында, Еуразия құрлығының орталығында орналасқан қала. 
Бұл қаланы «қала-бақ» деп те атайды.
"""
question = "Алматы қаласы Қазақстанның қай бөлігінде орналасқан?"

# Tokenize the input
inputs = tokenizer.encode_plus(
    question, 
    context, 
    add_special_tokens=True, 
    return_tensors="pt"
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Perform inference
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

# Find the answer's start and end position
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits)

# Decode the answer from the context
answer = tokenizer.decode(input_ids[0][start_index:end_index + 1])

print(f"Question: {question}")
print(f"Answer: {answer}")
