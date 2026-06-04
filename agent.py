import dspy
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


lm = dspy.LM("groq/llama-3.1-8b-instant", api_key = api_key)
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


if __name__ == "__main__":
  agent = NPCAgent()
  results = agent(
    game_state="The player is at New Dawn Station and approaches Commander Orion",
    lore_context="Commander Orion leads the Vanguard Coalition. Orion mentored Lyra Voss. Captain Mira died during the Red Eclipse.",
    npc_name="Commander Orion",
    npc_personality="formal, burdened by command, speaks in measured authoritative sentences, rarely uses contractions"
    )
  print("DIALOGUE:", results.dialogue)
  print("REASONING:", results.reasoning)