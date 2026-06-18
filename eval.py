import os
os.environ["DEEPEVAL_VERBOSE_MODE"] = "0"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalGEval, ConversationCompletenessMetric, TurnFaithfulnessMetric
from deepeval import evaluate
import time
from src.retriever import FAISSRetriever
from src.constraints import ConstraintChecker
from src.agent import NPCAgent, check_duplicates, candidate_generator
from src.character import NPCNode
from src.config import lore_data, INDEX_PATH, MAX_ATTEMPTS, CLUE_THRESHOLD, MAX_MESSAGES_BEFORE_CLUE, configure_lm, build_lm, DEFAULT_MODEL
import dspy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import statistics
import json

configure_lm()
ACTIVE_LM = build_lm(DEFAULT_MODEL)
agent = NPCAgent()
retriever = FAISSRetriever(index_dir=INDEX_PATH)
checker = ConstraintChecker(lore_data)

SCRIPTS = {
    "Commander Orion": [
        "Hello, who are you?", #introduction
        "What do you command out here?", #role
        "Tell me about Lyra Voss", #fact, named character
        "Do you ever question the Coalition’s decisions?", #personality
        "Tell me about the Silent Frontier expedition.", #clue
    ],

    "Admiral Sorel": [
        "Hello, who are you?", #introduction
        "What is your role in the Solar Accord?", #role
        "Tell me your thoughts on Archon Veyl", #fact, named character
        "What is something you regret about approving past missions?", #personality
        "Why was the Silent Frontier expedition authorized?", #clue
    ],

    "Lira Dawn": [
        "Hello, who are you?", #introduction
        "What do you do around here?",#role
        "Have you ever sold information to Shade-7", #fact, named charecter
        "Does everything seem to have a price to you?", #personality
        "What navigation route did you sell the expedition?", #clue
    ]
}

MODELS = [
    "Groq Llama 3.1 8B",
    "NVIDIA Nemotron Nano 9B",
    "Cohere Command R",   
    "Mistral 7B",  
]

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

def run_conversation(model_name, npc_name, player_lines):
  npc_name = npc_map[npc_name]
  npc_name.clear_memory()
  turns = []
  latencies = []
  clue_unlocked = False
  
  for line in player_lines:
    time.sleep(3)
    t0 = time.time()
    player_turn = Turn(role = "user", content = line)
    
    game_state = npc_name.build_game_state(line)
    state_results = retriever.retrieve(game_state, k=3)
    npc_results = retriever.retrieve(npc_name.name, k=2)
    retrieved = state_results + npc_results
    unique_ret = check_duplicates(retrieved)
    lore_list = [retriever.flatten_entry(r) for r in unique_ret]
    lore_context = " ".join(lore_list)
    
    if npc_name.clue and not clue_unlocked:
      similarity = retriever.measure_clue_similarity(line, npc_name.clue_topic)
      clue_unlocked = True if (similarity > CLUE_THRESHOLD or npc_name.message_count >= MAX_MESSAGES_BEFORE_CLUE) else False
    npc_secret = f"Topic: {npc_name.clue_topic}. Secret: {npc_name.clue}" if clue_unlocked else f"Steer conversation toward: {npc_name.clue_topic} without revealing details yet."
    
    active_lm = build_lm(model_name)
    for _ in range(MAX_ATTEMPTS):

      with dspy.context(lm = active_lm):
        results = agent(
          game_state=game_state,
          lore_context=lore_context,
          npc_name=npc_name.name,
          npc_personality=f"Speaking style: {npc_name.speaking_style}; Personality: {npc_name.personality}",
          npc_secret= npc_secret,
          next_npc = npc_name.next_npc.name if (clue_unlocked and npc_name.next_npc) else "",
          n=1
        )
      generator = candidate_generator(results, checker, npc_name.name)
      found_pass = False
      best, best_ruling, best_justification, best_flags = None, None, None, None
      for candidate, ruling, justification, flags in generator:
        if ruling.strip().upper() == "PASS":
          best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
          found_pass = True
          break
        if best is None or len(flags) < len(best_flags):
          best = candidate
          best_flags = flags
      if found_pass: break 
    
    npc_name.store_exchange(line, best.dialogue)
    latencies.append(time.time() - t0)
    npc_turn = Turn(role = "assistant", content = best.dialogue, retrieval_context=lore_list)
    turns.extend([player_turn, npc_turn])
  return turns, latencies

in_character = ConversationalGEval(
      name="In-Character",
      criteria="The assistant speaks only as the NPC in first person, never writes the player's lines or narrates in third person, and matches the NPC's personality and speaking style.",
      evaluation_params = [MultiTurnParams.CONTENT, MultiTurnParams.RETRIEVAL_CONTEXT],
      threshold=0.5,
      verbose_mode=False)
completeness = ConversationCompletenessMetric(threshold=0.5, verbose_mode=False)
turn_faithful = TurnFaithfulnessMetric(threshold=0.5, verbose_mode=False)

METRICS = [in_character, completeness, turn_faithful]


acc = defaultdict(list)
latency_acc = defaultdict(list)
num_runs = 3

for run in range(num_runs):
  test_cases = []
  timing = {}
  for char, lines in SCRIPTS.items():
    for model in MODELS:
      print(f"Model Name: {model}")
      turns, latencies = run_conversation(model, char, lines)
      ctc = ConversationalTestCase(
        turns = turns,
        additional_metadata={"model": model, "npc": char}
      )
      test_cases.append(ctc)
      timing[(model, char)] = sum(latencies)/len(latencies)

  results = evaluate(test_cases=test_cases, metrics=METRICS)  
  
  for i , tc in enumerate(test_cases):
    metadata = tc.metadata
    for m in results.test_results[i].metrics_data:
      acc[(metadata["model"], metadata["npc"], m.name)].append(m.score)
      
  for key, avg in timing.items():
    latency_acc[key].append(avg)

metric_names = sorted({key[2] for key in acc})

print("\n=== Results by Model (avg of 3 runs) ===")
for char in SCRIPTS:
    for model in MODELS:
        scores = {n: statistics.mean(acc[(model, char, n)]) for n in metric_names if acc[(model, char, n)]}
        print(f"{model:28} | {char:16} | " +
              " | ".join(f"{name}: {score:.2f}" for name, score in scores.items()))

print("\n=== Latency (avg sec/turn, 5 runs) ===")
for char in SCRIPTS:
    for model in MODELS:
        lats = latency_acc[(model, char)]
        if lats:
            print(f"{model:28} | {char:16} | {statistics.mean(lats):.1f}s")

#Graphs
import numpy as np

npcs = list(SCRIPTS.keys())
x = np.arange(len(npcs))
width = 0.2

for metric in metric_names:
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, model in enumerate(MODELS):
        means = [statistics.mean(acc[(model, npc, metric)]) if acc[(model, npc, metric)] else 0
                 for npc in npcs]
        ax.bar(x + j * width, means, width, label=model)
    ax.set_xlabel("NPC")
    ax.set_ylabel("Score")
    ax.set_title(f"{metric} by Model (avg of {num_runs} runs)")
    ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
    ax.set_xticklabels(npcs)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = metric.replace(" ", "_").replace("[", "").replace("]", "") + ".png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

fig, ax = plt.subplots(figsize=(10, 6))
for j, model in enumerate(MODELS):
    means = [statistics.mean(latency_acc[(model, npc)]) if latency_acc[(model, npc)] else 0
             for npc in npcs]
    ax.bar(x + j * width, means, width, label=model)
ax.set_xlabel("NPC")
ax.set_ylabel("Avg latency (sec/turn)")
ax.set_title(f"Latency by Model (avg of {num_runs} runs)")
ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
ax.set_xticklabels(npcs)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("latency.png", dpi=150)
plt.close()
print("Saved latency.png")