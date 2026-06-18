from .config import lore_data
from .memory import create_npc_table, store_message, get_recent_history, get_message_count, clear_table

class NPCNode:
    def __init__(self, clue, char_data, prerequisites=None):
        self.name = char_data['name']
        self.faction = char_data.get('faction')
        self.personality = char_data.get('personality')
        self.speaking_style = char_data.get('speaking_style')
        self.alive = char_data.get('alive')
        self.prerequisites = prerequisites
        self.message_count = 0
        self.clue_topic = None
        self.clue = clue
        self.next_npc = None #linked list for progression
        self.db_table = f"memory_{self.name.replace(' ', '_').replace('-', '_').lower()}"
        create_npc_table(self.db_table) #creates table on instantiation
        
    def store_exchange(self, player_message, npc_message):
        store_message(self.db_table, "player", player_message)
        store_message(self.db_table, "npc", npc_message)
        self.message_count = get_message_count(self.db_table)
        
    def get_history(self, n):
        return get_recent_history(self.db_table, n = n)
    
    def build_game_state(self, player_input, location = "New Dawn Station"):
        history = self.get_history(n = 3)
        history_str = "\n".join([f"{role}: {msg}" for role, msg in history])
        return f"Location: {location}. Speaking with {self.name}.\nRecent conversation:\n{history_str}\nPlayer says: {player_input}"
    
    def clear_memory(self):
        return clear_table(self.db_table)
    
    @classmethod #pulls out info for json, makes char Node from pulled information
    def from_lore(cls, lore_data, npc_name):
      char_data = {}
      clue = ""
      for c in lore_data['characters']:
          if c['name'] == npc_name:
              char_data = c
      for m in lore_data['mystery_clues']:
          if m['source'] == npc_name:
              clue = m['clue']
      return cls(clue=clue, char_data=char_data)