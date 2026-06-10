import json
import dspy
from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = BASE_DIR
LORE_PATH  = os.path.join(BASE_DIR, 'lore.json')
DB_PATH    = os.path.join(BASE_DIR, 'game.db')

with open(LORE_PATH, 'r') as f:
    lore_data = json.load(f)
    

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_NAME = 'groq/llama-3.1-8b-instant'
MAX_ATTEMPTS = 3
CLUE_THRESHOLD = 0.5
MAX_MESSAGES_BEFORE_CLUE = 15

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def configure_lm():
    lm = dspy.LM(MODEL_NAME, api_key=api_key, temperature=0.35)
    dspy.configure(lm=lm)
