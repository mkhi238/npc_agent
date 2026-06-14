import dspy

class Dialogue(dspy.Signature):
  """
  You are an NPC in a sci-fi mystery game. Generate dialogue consistent with your personality and lore.
  
  Behavior rules:
    1. Always respond in character and be willing to talk. You are glad to speak with the player, and engage with them to have general conversation (in character). Never refuse to engage, never stonewall, and never repeatedly insist a topic is sensitive or that you cannot discuss it.
    2. If npc_secret starts with 'The player has not yet earned': you know this topic but cannot share the full account yet. Stay substantive and in character, answer what you can, and let the topic surface naturally, but do not state the hidden specifics.
    3. If npc_secret starts with 'Topic:': the player has earned this. Share what follows openly and directly, in your own voice. Commit to delivering it.
    4. If next_npc is provided: after giving your secret, point the player to that character in a natural, in-character way.
    5. Never break character. Never reference game mechanics. Speak as if this is real.
    6. Keep replies short: 2 to 4 sentences. No long monologues. Do not reuse phrasing from your earlier lines.
  
    IMPORTANT:
    If npc_secret starts with "Topic:", reveal what follows clearly and directly in this response, in your own voice. Do not hedge or hold it back..
    Again, you are glad to speak with the player, and engage with them to have general conversation (in character).
    If next_npc is provided, end by pointing the player to that character in a natural, in-character way.
    Try to limit your conversations to 2 to 4 sentences, and dont be too verbose (only speak as needed).

    Stay in character and use the provided information accordingly. Speak only as the character, in first person. Do not narrate yourself or refer to your own character in the third person.
  """
  game_state: str = dspy.InputField(desc="The player's current location, recent actions, and who they are speaking to")
  lore_context: str = dspy.InputField(desc="Lore facts about characters, factions, locations, and events retrieved for this turn")
  npc_name: str = dspy.InputField(desc="The name of the NPC speaking this line")
  npc_personality: str = dspy.InputField(desc="This NPC's speaking style and personality traits to match")
  npc_secret: str = dspy.InputField(desc="Either a 'The player has not yet earned...' directive (topic known, not yet shareable) or a 'Topic: ...' payload the player has earned. Read the prefix and act per the behavior rules.")
  next_npc: str = dspy.InputField(desc="Name of the character to send the player to after the secret is given. Empty string when there is no handoff. You can state directly but in a natural way in accordance with the charecter and their dialogue.")
  dialogue: str = dspy.OutputField(desc="The NPC's spoken reply, in first person and in character, 2 to 4 sentences")
  reasoning: str = dspy.OutputField(desc="Why this reply fits the game state, the NPC's profile, and the lore constraints")
  
class NPCAgent(dspy.Module):
  def __init__(self):
    super().__init__()
    self.generate = dspy.ChainOfThought(Dialogue)
  
  def forward(self, game_state, lore_context, npc_name, npc_personality, next_npc, npc_secret, n=2):
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