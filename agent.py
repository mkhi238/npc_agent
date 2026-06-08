import dspy

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
  
  def forward(self, game_state, lore_context, npc_name, npc_personality, n=3):
    candidates = []
    for _ in range(n):
      result = self.generate(game_state=game_state, 
                            lore_context=lore_context, 
                            npc_name = npc_name, 
                            npc_personality = npc_personality)
      candidates.append(result)
    return candidates

def check_duplicates(retr):
  seen = set()
  unique_retreieved = []
  for i in retr:
    if i['id'] not in seen:
      seen.add(i['id'])
      unique_retreieved.append(i)
  return unique_retreieved

def candidate_generator(candidates, checker, npc_name):
  for c in candidates:
    ruling, justification, flags = checker.check(c.dialogue, npc_name)
    yield c, ruling, justification, flags