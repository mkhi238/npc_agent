import json

with open('/mnt/d/npc_agent/lore.json', 'r') as f:
    lore_data = json.load(f)

LORE_PATH = '/mnt/d/npc_agent/lore.json'
INDEX_PATH = '/mnt/d/npc_agent'
DB_PATH = '/mnt/d/npc_agent/game.db'
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_NAME = 'groq/llama-3.1-8b-instant'
MAX_ATTEMPTS = 3
CLUE_THRESHOLD = 0.5
MAX_MESSAGES_BEFORE_CLUE = 15
