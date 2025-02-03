import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import struct

# Step 1: Read the input file
#file_path = './../libcell_properties.csv'  # Adjust the path to your file
file_path = 'ASAP7_libcells.txt' #'nangate45_libcells.txt' #'ASAP7_libcells.txt' # 'nangate_libcells.txt'
if file_path.endswith('.csv'):
    libcell_properties = pd.read_csv(file_path)
    libcell_names = libcell_properties['libcell_name'].dropna().unique()  # Remove NaN and get unique values
elif file_path.endswith('.txt'):
    with open(file_path, 'r') as f:
        libcell_names = list(set(line.strip() for line in f if line.strip()))  # Remove duplicates and empty lines
else:
    raise ValueError("Unsupported file format. Please provide a .csv or .txt file.")

# Step 2: Sort the libcell_name column strings
libcell_names_sorted = sorted(libcell_names)

# Step 3: Load the DeBERTa-v3-base tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = AutoModel.from_pretrained('microsoft/deberta-v3-base')

# Step 4: Generate embeddings
embeddings = []
embedding_size = None
for name in libcell_names_sorted:
    inputs = tokenizer(name, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Pool the embeddings
    embeddings.append(embedding)
    if embedding_size is None:
        embedding_size = embedding.shape[0]

# Step 5: Save embeddings as a binary file
output_file = 'libcell_embeddings.bin'
with open(output_file, 'wb') as f:
    # Write the number of items and the embedding size
    f.write(struct.pack('Q', len(libcell_names_sorted)))  # size_t for num_items
    f.write(struct.pack('Q', embedding_size))  # size_t for embedding_size

    # Write each embedding in order
    for embedding in embeddings:
        f.write(struct.pack(f'{embedding_size}f', *embedding))

print(f"Embeddings saved to {output_file}.")

# Print a sample of the embeddings to verify
print("Sample of the embeddings:")
for i in range(5):
    print(f"{libcell_names_sorted[i]}: {embeddings[i][:5]}")
