import dspy
import os
from dotenv import load_dotenv
from retriever import FAISSRetriever
from constraints import ConstraintChecker
from agent import NPCAgent, check_duplicates, candidate_generator
from config import lore_data, MODEL_NAME, INDEX_PATH

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq/llama-3.1-8b-instant"

lm = dspy.LM(MODEL_NAME, api_key = api_key)
dspy.configure(lm = lm)

agent = NPCAgent()
retriever = FAISSRetriever(index_dir="/mnt/d/npc_agent")

player_input = input("What do you say to the NPC? > ")
game_state = f"The player is at New Dawn Station and approaches Commander Orion. Player says: {player_input}"
npc_name = "Commander Orion"

state_results = retriever.retrieve(game_state, k=3)
npc_results = retriever.retrieve(npc_name, k=2)
retrieved = state_results + npc_results

unique_ret = check_duplicates(retrieved)
lore_context = " ".join([retriever.flatten_entry(r) for r in unique_ret])
print(lore_context)

npc_data = retriever.characters.get(npc_name)
npc_personality = npc_data['speaking_style']

results = agent(
    game_state=game_state,
    lore_context=lore_context,
    npc_name=npc_name,
    npc_personality=npc_personality
)

generator = candidate_generator(results, ConstraintChecker(lore_data), npc_name)
best, best_ruling, best_justification, best_flags = None, None, None, None
#lazy evaluation on candidate strings
for candidate, ruling, justification, flags in generator:
  if ruling.strip().upper() == "PASS":
    best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
    break
  if best is None or len(flags) < len(best_flags):
    best = candidate
    best_ruling = ruling
    best_justification = justification
    best_flags = flags
print("DIALOGUE:", best.dialogue)
print("REASONING:", best.reasoning)
print("RULING:", best_ruling)
print("JUSTIFICATION:", best_justification)
print("FLAGS:", best_flags)