import dspy
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq/llama-3.1-8b-instant"

lm = dspy.LM(MODEL_NAME, api_key = api_key)
dspy.configure(lm = lm)

class Ruling(dspy.Signature):
    """ 
    You are a strict lore consistency judge for a sci-fi game world. 
    You will be given NPC dialogue, any lore warnings flagged by a rule-based checker, 
    and the character's established profile. Your job is to determine if the dialogue 
    is consistent with the established lore and the charecter's established profile. Be strict -- if a flag suggests a potential 
    violation, investigate it carefully. Return PASS only if you are confident the 
    dialogue is lore-consistent. Return FAIL with a specific reason if not.
    """
    dialogue: str = dspy.InputField(desc="The NPC dialogue to evaluate")
    flags: str = dspy.InputField(desc="Lore consistency warnings flagged by the hard checker")
    char_profile: str = dspy.InputField(desc="The NPC's established personality, faction, and speaking style")
    ruling: str = dspy.OutputField(desc="PASS if dialogue is lore-consistent, FAIL with specific reason if not")

class JudgeAgent(dspy.Module):
  def __init__(self):
    self.generate = dspy.ChainOfThought(Ruling)
    
  def forward(self, dialouge_str, flags, char_profile):
    result = self.generate(dialogue = dialouge_str,
                           flags = flags,
                           char_profile = char_profile)
    return result
  
class ConstraintChecker:
  #constraint checker takes in results.dialouge and the FAISS retriever
  def __init__(self, lore_data):
    self.lore_data = lore_data
    self.dead_characters = {c['name'] for c in lore_data['characters'] if c['alive'] == False}
    self.alive_characters = {c['name'] for c in lore_data['characters'] if c['alive'] == True}
    self.destroyed_locations = {l['name'] for l in lore_data['locations'] if l['exists'] == False}
    self.uncontrolled_locations = {l['name'] for l in lore_data['locations'] if l['controlled_by'] is None}
    self.artifact_possessors = {a['name']: a['possessor'] for a in lore_data['artifacts'] if a['possessor']}
    self.unpossessed_artifacts = {a['name'] for a in lore_data['artifacts'] if a['possessor'] is None}
    self.artifact_unknown_loc = {a['name'] for a in lore_data['artifacts'] if a['location'] is None}
    self.faction_enemies = {f['name']: f['enemies'] for f in lore_data['factions']}
    self.faction_allies = {f['name']: f['allies'] for f in lore_data['factions']}
    self.character_factions = {c['name']: c['faction'] for c in lore_data['characters']}
    self.character_lookup = {c['name']: c for c in lore_data['characters']}
  
  
  def flag_exceptions(self, dialogue):
      flags = []
      
      # Dead characters
      for name in self.dead_characters:
          if name in dialogue:
              flags.append(f"WARNING: '{name}' appears in dialogue but is dead. Ensure they are not referenced as living.")

      # Destroyed locations
      for loc in self.destroyed_locations:
          if loc in dialogue:
              flags.append(f"WARNING: '{loc}' appears in dialogue but was destroyed and no longer exists. Ensure it is not referenced as a reachable location.")

      # Uncontrolled locations
      for loc in self.uncontrolled_locations:
          if loc in dialogue:
              flags.append(f"WARNING: '{loc}' appears in dialogue but has no controlling faction. Ensure it is not referenced as under any faction's control.")

      # Artifact possessors
      for artifact, possessor in self.artifact_possessors.items():
          if artifact in dialogue:
              flags.append(f"WARNING: '{artifact}' appears in dialogue. It is currently possessed by {possessor}. Ensure possession is referenced correctly.")

      # Unpossessed artifacts
      for artifact in self.unpossessed_artifacts:
          if artifact in dialogue:
              flags.append(f"WARNING: '{artifact}' appears in dialogue but has no current possessor. Ensure it is not referenced as held by any character.")

      # Artifacts with unknown location
      for artifact in self.artifact_unknown_loc:
          if artifact in dialogue:
              flags.append(f"WARNING: '{artifact}' appears in dialogue but its location is unknown. Ensure it is not referenced as being in a specific place.")

      # Faction enemies
      for faction, enemies in self.faction_enemies.items():
          if faction in dialogue:
              for enemy in enemies:
                  if enemy in dialogue:
                      flags.append(f"WARNING: '{faction}' and '{enemy}' both appear in dialogue but are enemies. Ensure they are not referenced as allied or friendly.")

      # Faction allies
      for faction, allies in self.faction_allies.items():
          if faction in dialogue:
              for ally in allies:
                  if ally in dialogue:
                      flags.append(f"NOTE: '{faction}' and '{ally}' both appear in dialogue and are allies. Ensure their relationship is portrayed correctly.")

      # Character faction allegiance
      for char, faction in self.character_factions.items():
          if char in dialogue and faction and faction in dialogue:
              flags.append(f"WARNING: '{char}' appears alongside their faction '{faction}'. Ensure their allegiance is portrayed correctly.")

      # Alive characters (sanity check — flag if referenced in a death context)
      for name in self.alive_characters:
          if name in dialogue:
              if f"{name} died" in dialogue or f"{name} is dead" in dialogue:
                  flags.append(f"WARNING: '{name}' appears to be referenced as dead but is alive.")


      return flags
    
  def get_char_profile(self, npc_name):
    char = self.character_lookup.get(npc_name, {})
    return f"{npc_name}: faction={char.get('faction')}, personality={char.get('personality')}, speaking_style={char.get('speaking_style')}, alive={char.get('alive')}"
  
  def llm_check(self, dialouge, flags, char_profile):
    flags_str = "\n".join(flags) if flags else "No flags raised."
    judge = JudgeAgent()
    result = judge(
      dialouge_str = dialouge,
      flags = flags_str,
      char_profile = char_profile
    )
    return result.ruling
  
  def check(self, dialouge, npc_name):
    flags = self.flag_exceptions(dialouge)
    char_profile = self.get_char_profile(npc_name)
    ruling = self.llm_check(dialouge, flags, char_profile)
    return ruling, flags
    
  
if __name__ == "__main__":
  import json
  with open('/mnt/d/npc_agent/lore.json', 'r') as f:
      lore_data = json.load(f)
  
  checker = ConstraintChecker(lore_data)
  
  test_dialogue = "Captain Mira, who is dead, will meet you at New Dawn Station. She has been expecting your arrival."
  ruling, flags = checker.check(test_dialogue, "Commander Orion")
  
  print("FLAGS:", flags)
  print("RULING:", ruling)  
  