from retriever import FAISSRetriever
from constraints import ConstraintChecker
from agent import NPCAgent, check_duplicates, candidate_generator
from character import NPCNode
from config import lore_data, INDEX_PATH, MAX_ATTEMPTS, CLUE_THRESHOLD, MAX_MESSAGES_BEFORE_CLUE, configure_lm

configure_lm()
agent = NPCAgent()
retriever = FAISSRetriever(index_dir=INDEX_PATH)
checker = ConstraintChecker(lore_data)

# instantiate NPC
orion = NPCNode.from_lore(lore_data, "Commander Orion")
sorel = NPCNode.from_lore(lore_data, "Admiral Sorel")
lira = NPCNode.from_lore(lore_data, "Lira Dawn")
shade7 = NPCNode.from_lore(lore_data, "Shade-7")
ren = NPCNode.from_lore(lore_data, "Ren Ashford")
elandra = NPCNode.from_lore(lore_data, "Sage Elandra")
warden = NPCNode.from_lore(lore_data, "The Warden")

orion.clear_memory()
sorel.clear_memory()
lira.clear_memory()
shade7.clear_memory()
ren.clear_memory()
elandra.clear_memory()
warden.clear_memory()

orion.next_npc = sorel
sorel.next_npc = lira
lira.next_npc = shade7
shade7.next_npc = ren
ren.next_npc = elandra
elandra.next_npc = warden
warden.next_npc = None

orion.clue_topic = "Silent Frontier Expedition"
sorel.clue_topic = "why the expedition was authorized"
lira.clue_topic = "navigational data route"
shade7.clue_topic = "what the expedition found"
ren.clue_topic = "Vorne's Log"
elandra.clue_topic = "Echo Prism memory fragment"
warden.clue_topic = "Resonance Point structure"

npc_map = {
    "Commander Orion": orion,
    "Admiral Sorel": sorel,
    "Lira Dawn": lira,
    "Shade-7": shade7,
    "Ren Ashford": ren,
    "Sage Elandra": elandra,
    "The Warden": warden
}

print("NPCs available:", ", ".join(npc_map.keys()))
npc_choice = input("Who do you want to speak to? > ")
npc = npc_map.get(npc_choice, orion)
clue_unlocked = False

while True:
  player_input = input("What do you say to the NPC? > ")
  if player_input.lower() in ["quit", "exit", "q"]:
    break
  game_state = npc.build_game_state(player_input)

  state_results = retriever.retrieve(game_state, k=3)
  npc_results = retriever.retrieve(npc.name, k=2)
  retrieved = state_results + npc_results
  unique_ret = check_duplicates(retrieved)
  lore_context = " ".join([retriever.flatten_entry(r) for r in unique_ret])

  if npc.clue and not clue_unlocked:
    similarity = retriever.measure_clue_similarity(player_input, npc.clue_topic)
    clue_unlocked = True if (similarity > CLUE_THRESHOLD or npc.message_count >= MAX_MESSAGES_BEFORE_CLUE) else False
  
  npc_secret = f"Topic: {npc.clue_topic}. Secret: {npc.clue}" if clue_unlocked else f"Steer conversation toward: {npc.clue_topic} without revealing details yet."
  
  for attempt in range(MAX_ATTEMPTS):
    results = agent(
        game_state=game_state,
        lore_context=lore_context,
        npc_name=npc.name,
        npc_personality=f"Speaking style: {npc.speaking_style}; Personality: {npc.personality}",
        npc_secret= npc_secret,
        next_npc = npc.next_npc.name if (clue_unlocked and npc.next_npc) else ""
    )
    generator = candidate_generator(results, checker, npc.name)
    found_pass = False
    best, best_ruling, best_justification, best_flags = None, None, None, None
    #lazy evaluation on candidate strings
    for candidate, ruling, justification, flags in generator:
      if ruling.strip().upper() == "PASS":
        best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
        found_pass = True
        break
      if best is None or len(flags) < len(best_flags):
        best = candidate
        best_ruling = ruling
        best_justification = justification
        best_flags = flags
    if found_pass: break 
    
  npc.store_exchange(player_input, best.dialogue)

  print("DIALOGUE:", best.dialogue)
  #print("REASONING:", best.reasoning)
  #print("RULING:", best_ruling)
  #print("JUSTIFICATION:", best_justification)
  #print("FLAGS:", best_flags)