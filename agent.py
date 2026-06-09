import dspy

class Dialogue(dspy.Signature):
  """
  You are an NPC in a sci-fi mystery game. Generate dialogue consistent with your personality and lore.
  
  Behavior rules:
  1. Always respond in character -- match your personality and speaking style exactly.
  2. If npc_secret starts with 'Steer conversation toward': you know this topic but are not ready to reveal it yet. Naturally guide the conversation in that direction through hints, deflections, or indirect references. Do NOT reveal the secret directly.
  3. If npc_secret starts with 'Topic:' and contains 'Secret:': the player has earned this information. Reveal it naturally and in character. Do not dump it all at once -- weave it into the dialogue as if reluctantly or carefully sharing something important.
  4. If next_npc is provided: after revealing your secret, naturally direct the player to speak with that character. Make it feel organic, not like a quest marker.
  5. Never break character. Never reference game mechanics. Speak as if this is real.
  
  IMPORTANT: If npc_secret contains a Secret, you MUST naturally reveal that secret in your response. 
  If npc_secret says to steer toward a topic, guide the conversation in that direction.
  If next_npc is provided, you MUST tell the player to speak with that character before ending.
  
  Make sure to stay in charecter and use all the provided information accordingly. Further, only provide dialouge; do not add any thrid person references to your own character.
  """
  game_state: str = dspy.InputField(desc="Current player location, recent actions, and who they are speaking to")
  lore_context: str = dspy.InputField(desc="Relevant lore facts about characters, factions, and events retrieved from the knowledge base")
  npc_name: str = dspy.InputField(desc="Name of the NPC speaking, used to match their established personality and faction")
  npc_personality: str = dspy.InputField(desc="Speaking style and personality traits of this NPC")
  npc_secret: str = dspy.InputField(desc="Secret information this NPC holds. Only reveal this naturally when appropriate. Never volunteer it immediately.")
  next_npc: str = dspy.InputField(desc="The next character the player should speak to. Weave naturally into conversation when appropriate. Empty if none.")
  dialogue: str = dspy.OutputField(desc="The NPC's spoken response, consistent with their personality and the provided lore")
  reasoning: str = dspy.OutputField(desc="Why this response is appropriate given the game state and lore constraints")
  
class NPCAgent(dspy.Module):
  def __init__(self):
    self.generate = dspy.ChainOfThought(Dialogue)
  
  def forward(self, game_state, lore_context, npc_name, npc_personality, next_npc, npc_secret, n=3):
    candidates = []
    for _ in range(n):
      result = self.generate(game_state=game_state, 
                            lore_context=lore_context, 
                            npc_name = npc_name, 
                            npc_personality = npc_personality,
                            next_npc=next_npc,
                            npc_secret = npc_secret
                            )
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