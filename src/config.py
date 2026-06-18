import json
import dspy
from dotenv import load_dotenv
import os
from groq import Groq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = BASE_DIR
LORE_PATH  = os.path.join(BASE_DIR, 'lore.json')
DB_PATH    = os.path.join(BASE_DIR, 'game.db')

with open(LORE_PATH, 'r') as f:
    lore_data = json.load(f)
    

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_NAME = 'groq/llama-3.1-8b-instant'
MAX_ATTEMPTS = 1
CLUE_THRESHOLD = 0.5
MAX_MESSAGES_BEFORE_CLUE = 15


TTS_MODEL = "canopylabs/orpheus-v1-english"
CHAR_LIMIT = 200

# Selectable text models. Default is Groq llama; NVIDIA Nemotron is the sponsor option.
# Each entry is what configure_lm() needs to build the right dspy.LM.
MODEL_REGISTRY = {
    "Groq Llama 3.1 8B": {
        "model": MODEL_NAME,              # litellm groq/ prefix already in MODEL_NAME
        "api_key": api_key,
        "api_base": None,
    },
    "NVIDIA Nemotron Nano 9B": {
        "model": "openai/nvidia/nvidia-nemotron-nano-9b-v2",
        "api_key": nvidia_api_key,
        "api_base": "https://integrate.api.nvidia.com/v1",
    },
    "Cohere Command R7B": {
    "model": "cohere/command-r7b-12-2024",
    "api_key": cohere_api_key,
    "api_base": None,
    },
    "Mistral 7B": {
    "model": "mistral/open-mistral-7b",
    "api_key": mistral_api_key,
    "api_base": None,
}
}

DEFAULT_MODEL = "Groq Llama 3.1 8B"

def build_lm(choice=DEFAULT_MODEL):
    """Build (but do not globally configure) a dspy.LM for the chosen model."""
    cfg = MODEL_REGISTRY.get(choice, MODEL_REGISTRY[DEFAULT_MODEL])
    kwargs = {"temperature": 0.35}
    if cfg["api_key"]:
        kwargs["api_key"] = cfg["api_key"]
    if cfg["api_base"]:
        kwargs["api_base"] = cfg["api_base"]
    return dspy.LM(cfg["model"], **kwargs)


def configure_lm(choice=DEFAULT_MODEL):
    """Configure DSPy's global LM from the registry (call once, on main thread)."""
    dspy.configure(lm=build_lm(choice))
    return choice
    
def configure_tts():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))
