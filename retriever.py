from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import lore_data as data, EMBEDDING_MODEL

  
class FAISSRetriever:
  def __init__(self, index_dir, model_name = EMBEDDING_MODEL):
    self.index_dir = Path(index_dir)
    self.model_name = model_name
    
    self.model = SentenceTransformer(model_name, device = 'cpu')
    self.texts = []
    self.metadata = []
    for category, entries in data.items():
      for jsons in entries:
        text = self.flatten_entry(jsons)
        self.texts.append(text)
        self.metadata.append(jsons)
    self.embeddings = self.model.encode(self.texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    dim = self.embeddings.shape[1]
    self.index = faiss.IndexFlatIP(dim)
    self.index.add(self.embeddings)
    self.characters = {c['name']: c for c in data['characters']}
    
    #function to make the json one string
  def flatten_entry(self, entry):
    fields = ['name', 'faction', 'fact', 'personality', 'speaking_style', 
            'relationship', 'outcome', 'controlled_by', 'possessor', 'location']
    
    char_profile = []
    for f in fields:
      value = entry.get(f)
      if value and value is not None:
        char_profile.append(str(value))
      else:
        continue
    return ' '.join(char_profile)
  
  def encode_query(self, text):
    vector = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    vector = vector.astype(np.float32)
    return vector
  
  def retrieve(self, query, k=3):
    query_vector = self.encode_query(query)
    _, idx = self.index.search(query_vector, k)
    metadata_queries = []
    for i in idx[0]:
      metadata_queries.append(self.metadata[i])
    return metadata_queries
