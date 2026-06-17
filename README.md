# 🛰️ Silent Frontier

#### What it is in one line: 
An NPC dialogue reasoning agent. Interrogate a chain of seven characters across a sci-fi mystery, earn each one's clue, and follow the trail from Commander Orion to the Warden. 

#### Core tech stack in one line: 
A neurosymbolic NPC dialogue agent that keeps four sub-9B models in character and lore-faithful using a DSPy reasoning pipeline and a rule+LLM constraint checker.

## Demo links

## Objective & Gameplay
The objective of the game is to uncover the truth behind the Silent Frontier Expedition. You begin with Commander Orion and must earn each NPC's clue to learn who to question next, working down a chain of seven characters until you reach the Warden and piece together what really happened. 

The player does so by prompting each character for a response in free-form conversation. Every NPC guards a secret, and stays evasive (or will awnser your questions) until your queries steer close enough to it. Ask the right questions, and the clue unlocks, revealing both a piece of the mystery and the name of the next person to find. A player can see how close their query is to the the clue through the QUERY MATCH mechanic located on the screen below the conversation window, and the game will tell the player if the clue is locked (STATUS: LOCKED) or unlocked (STATUS: CLUE UNLOCKED).

## ⚙️ How it works

<img width="1558" height="713" alt="userflow" src="https://github.com/user-attachments/assets/54e3f255-d61d-4711-b382-fb43969ee979" />


### Architecture 
The LLM agent begins by initializing each of the 7 named NPCs (Commander Orion, Admiral Sorel, Lira Dawn, Shade-7, Ren Ashford, Sage Elandra, and The Warden) and setting up their memory and clue states. Following this, the game state is built using the player input. The agent takes the player's input and uses a FAISS retriever to collect information related to the player's query from its contained lore (stored in lore.json). The player's query is then evaluated against the secret the NPC is guarding (always stored as a string) using a cosine similarity score. If the score is > 0.5, the clue is unlocked and the agent is provided with the clue's information, which was previously gated from it. If the score is < 0.5, the clue remains locked and the agent is not provided with that information. The agent is also given the last 3 exchanges between itself and the player, stored in a SQLite database.Following this, the agent builds the prompt. A generator creates a candidate response, which is then evaluated against a set of constraints. The check runs in two layers: rule-based flags catch hard lore violations (for example, referencing a dead character as alive, or a destroyed location as still existing), and an LLM judge evaluates the candidate against the constraints and returns a PASS or FAIL verdict. If a candidate fails, it is re-sent to the model, which is tasked with regenerating a new response. If no issues are raised, or a MAX_ATTEMPTS number of regeneration attempts has been reached, the candidate response (or the one with the fewest flags raised) is sent as the final response. The final response is stored back in the database alongside the player's prompt, ready for the next query.

### Models
The model used by the agent is selectable by the user, between 4 choices. (However, the game demo on the Hugging Face Space only allows 2 of them to be played, the other 2 are available when running locally.)

| Model | API Source | Parameter Count | Available on HF Space |
|---|---|---|---|
| Llama 3.1 8B Instant | Groq | 8B | ✅ Yes (default) |
| Nemotron Nano 9B v2 | NVIDIA | 9B | ✅ Yes |
| Command R7B | Cohere | 7B | ❌ Local only |
| Mistral 7B | Mistral | 7B | ❌ Local only |

All four models are under the hackathon's 32B parameter cap. The LLM judge in the constraint checker uses the Llama 3.1 8B model consistently, regardless of which model the agent uses for generation.

Built and extended on by **mkhi238** for the Hugging Face × Gradio Build Small hackathon.

