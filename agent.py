import dspy
import os
from dotenv import load_dotenv
from retriever import FAISSRetriever

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
INDEX_PATH = "/mnt/d/npc_agent"
MODEL_NAME = "groq/llama-3.1-8b-instant"

lm = dspy.LM(MODEL_NAME, api_key = api_key)
dspy.configure(lm = lm)

class Dialogue(dspy.Signature):
  """
  Create an NPC given the provided information on game state and lore
  """
  game_state: str = dspy.InputField(desc="Current player location, recent actions, and who they are speaking to")
  lore_context: str = dspy.InputField(desc="Relevant lore facts about characters, factions, and events retrieved from the knowledge base")
  npc_name: str = dspy.InputField(desc="Name of the NPC speaking, used to match their established personality and faction")
  npc_personality: str = dspy.InputField(desc="Speaking style and personality traits of this NPC")
  
  
    
  dialogue: str = dspy.OutputField(desc="The NPC's spoken response, consistent with their personality and the provided lore")
  reasoning: str = dspy.OutputField(desc="Why this response is appropriate given the game state and lore constraints")
  
class NPCAgent(dspy.Module):
  def __init__(self):
    self.generate = dspy.ChainOfThought(Dialogue)
  
  def forward(self, game_state, lore_context, npc_name, npc_personality):
    result = self.generate(game_state=game_state, 
                           lore_context=lore_context, 
                           npc_name = npc_name, 
                           npc_personality = npc_personality)
    return result

def check_duplicates(retr):
  seen = set()
  unique_retreieved = []
  for i in retr:
    if i['id'] not in seen:
      seen.add(i['id'])
      unique_retreieved.append(i)
  return unique_retreieved

if __name__ == "__main__":
    agent = NPCAgent()
    retriever = FAISSRetriever(index_dir="/mnt/d/npc_agent")
    
    game_state = "The player is at New Dawn Station and approaches Commander Orion"
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
    print("DIALOGUE:", results.dialogue)
    print("REASONING:", results.reasoning)