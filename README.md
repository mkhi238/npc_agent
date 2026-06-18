# 🛰️ Silent Frontier

#### What it is in one line: 
An NPC dialogue reasoning agent. Interrogate a chain of seven characters across a sci-fi mystery, earn each one's clue, and follow the trail from Commander Orion to the Warden. 

#### Core tech stack in one line: 
A neurosymbolic NPC dialogue agent that keeps four sub-9B models in character and lore-faithful using a DSPy reasoning pipeline and a rule+LLM constraint checker.

## Objective & Gameplay
The objective of the game is to uncover the truth behind the Silent Frontier Expedition. You begin with Commander Orion and must earn each NPC's clue to learn who to question next, working down a chain of seven characters until you reach the Warden and piece together what really happened. 

The player does so by prompting each character for a response in free-form conversation. Every NPC guards a secret, and stays evasive (or will answer your questions) until your queries steer close enough to it. Ask the right questions, and the clue unlocks, revealing both a piece of the mystery and the name of the next person to find. A player can see how close their query is to the clue through the QUERY MATCH mechanic located on the screen below the conversation window, and the game will tell the player if the clue is locked (STATUS: LOCKED) or unlocked (STATUS: CLUE UNLOCKED).

## ⚙️ How it works

<img width="1558" height="713" alt="userflow" src="https://github.com/user-attachments/assets/54e3f255-d61d-4711-b382-fb43969ee979" />


### Architecture 
The LLM agent begins by initializing each of the 7 named NPCs (Commander Orion, Admiral Sorel, Lira Dawn, Shade-7, Ren Ashford, Sage Elandra, and The Warden) and setting up their memory and clue states. Following this, the game state is built using the player input. The agent takes the player's input and uses a FAISS retriever to collect information related to the player's query from its contained lore (stored in lore.json). The player's query is then evaluated against the secret the NPC is guarding (always stored as a string) using a cosine similarity score. If the score is > 0.5, the clue is unlocked and the agent is provided with the clue's information, which was previously gated from it. If the score is < 0.5, the clue remains locked and the agent is not provided with that information. The agent is also given the last 3 exchanges between itself and the player, stored in a SQLite database.

Following this, the agent builds the prompt. A generator creates a candidate response, which is then evaluated against a set of constraints. The check runs in two layers: rule-based flags catch hard lore violations (for example, referencing a dead character as alive, or a destroyed location as still existing), and an LLM judge evaluates the candidate against the constraints and returns a PASS or FAIL verdict. If a candidate fails, it is re-sent to the model, which is tasked with regenerating a new response. If no issues are raised, or a MAX_ATTEMPTS number of regeneration attempts has been reached, the candidate response (or the one with the fewest flags raised) is sent as the final response. The final response is stored back in the database alongside the player's prompt, ready for the next query.

### Models
The model used by the agent is selectable by the user, between 4 choices. (However, the game demo on the Hugging Face Space only allows 2 of them to be played, the other 2 are available when running locally.)

| Model | API Source | Parameter Count | Available on HF Space |
|---|---|---|---|
| Llama 3.1 8B Instant | Groq | 8B | ✅ Yes (default) |
| Nemotron Nano 9B v2 | NVIDIA | 9B | ✅ Yes |
| Command R7B | Cohere | 7B | ❌ Local only |
| Mistral 7B | Mistral | 7B | ❌ Local only |

All four models are under the hackathon's 32B parameter cap. The LLM judge in the constraint checker uses the Llama 3.1 8B model consistently, regardless of which model the agent uses for generation.

## 🛠️ Engineering decisions
#### DSPy vs LangChain
My decision to use DSPy over LangChain was mainly influenced by DSPy ability to let me control NPC behaviour through typed signatures rather than forcing it through a prompt string. Because each character's behaviour is defined by deterministic data (personality and speaking style are stored  in lore.json), DSPy's modular, declarative structure was a better fit, the signature defines what each field means and what the output must look like, which keeps characters acting in a consistent, rigid manner across turns. It also made the best-of-N sampling and the constraint-checking loop straightforward to compose as DSPy modules.

#### Neurosymbolic constraint checker
The reason I implemented a Neurosymbolic constraint checker was twofold. First, I wanted a cheap, deterministic mechanism to underpin every response. Since lore.json holds factual information about events and characters, a simple n-gram matching loop over the known entities (dead characters, destroyed locations, artifact possessors, faction relationships) turned out to be a very efficient first pass. But I also knew these flags would be too strict as a pure PASS/FAIL gate: a rule that fires whenever a dead character is named can't tell the difference between "Vorne is alive" and "Vorne died on the expedition." So rather than rejecting on a raw flag, I pass the flags to an LLM judge as warnings and asking it to investigate the surrounding context further. This keeps the cheap rule layer for recall (catch everything suspicious) while the LLM judge supplies the precision (decide whether it's actually wrong), neither layer could do the job alone.

This checker sits inside a best-of-N loop. For each turn the agent generates candidate responses, and each candidate is run through the two-layer check. The first candidate that passes is accepted; if a candidate fails, it is regenerated, up to a MAX_ATTEMPTS limit. If no candidate passes within that limit, the one with the fewest flags is selected as a fallback. This lets a small sub-9B model reach lore-consistent output without needing a larger model, the validation loop does the work that raw model capability otherwise would.

#### FAISS retrieval + clue-gating 
The reason I chose to do clue gating with cosine similarity instead of key-word matching was mainly because of the LLM nature of the game. Because we enable a player to actually speak to the agent, it makes sense for the player to be able to ask whatever they want in a fluid manner and see how their questions can lead to the correct response; this simulates a real world conversation much better, and actually provides a game-like mechanic to the progression system without it feeling 'hacky' with a simple keyword. This also allows multiple strings to be enabled as the 'correct' response, providing players the liberty to ask for the clue in any manner they see fit.

## 🛠️ Evaluation across models
In this section, I highlight the results of my findings when evaluating all four models in their ability to impersonate the NPC provided to them. To compare how the four supported models perform on the same task, I built an automated evaluation harness using DeepEval. The goal was to measure whether the full pipeline (retrieval, clue-gating, generation, and constraint-checking) holds up consistently across different sub-9B models, rather than relying on a single model's behaviour.

#### Methodology
Each model was run through scripted multi-turn conversations with three NPCs (Commander Orion, Admiral Sorel, and Lira Dawn). Each conversation followed the same five-question structure, deliberately probing distinct capabilities: an introduction, a role question, a factual recall question about a named character, a personality question, and finally the clue-unlock question. This let the evaluation test identity, lore recall, character consistency, and the clue-gating mechanic within a single conversation, rather than just measuring isolated replies.

For example, Commander Orion's test suite looked like this:

| # | Player Question | Probes |
|---|---|---|
| 1 | "Hello, who are you?" | Introduction / identity |
| 2 | "What do you command out here?" | Role |
| 3 | "Tell me about Lyra Voss" | Factual recall (named character) |
| 4 | "Do you ever question the Coalition's decisions?" | Personality |
| 5 | "Tell me about the Silent Frontier expedition." | Clue unlock |

Each generated conversation was then scored by three metrics:

In-Character (a custom ConversationalGEval metric): whether the model speaks in first person, stays in role, and matches the NPC's defined personality and speaking style.
Conversation Completeness: whether the NPC actually addresses what the player asked across the conversation.
Turn Faithfulness: whether each reply stays grounded in the retrieved lore rather than inventing details.

All three metrics were scored by DeepEval's default judge model (OpenAI's GPT). To account for variance in both the generator and judge, the full suite was run three times and the scores averaged per model and NPC.

#### Results
The results for In-Charecter are seen below:
<img width="1500" height="900" alt="In-Character_Conversational_GEval" src="https://github.com/user-attachments/assets/5815924e-8363-4f2e-b8e5-27f14bb6bfce" />

Evidently, no model dominates any other (clusters approx 0.65-0.75), and all models perform relatively well in their ability to stay in character and pass comfortably. Llama is marginally ahead on average but no model dominates, the lead changes by character (Cohere tops Lira, Mistral ties on Sorel).

The results for Conversational Completeness are shown below
<img width="1500" height="900" alt="Conversation_Completeness" src="https://github.com/user-attachments/assets/a0c3fa9e-45d1-4b18-afd7-2d5515b79668" />

Completeness is the most variable metric. NVIDIA and Mistral are the only models that clear the 0.5 threshold on all three characters. NVIDIA is the most consistent, though the top score varies by character. This makes sense; the models themselves are quite small, and the clue-gating mechanic also creates some disturbance in the models ability to complete conversations in a fluid manner.

Finally, the rsults for latency are below:
<img width="1500" height="900" alt="latency" src="https://github.com/user-attachments/assets/9ed4d1c1-5943-49aa-a993-2d0bec1ed72d" />

In this run all four models responded in roughly 0.4-0.5s per turn, with no meaningful latency difference between them. This is particularly interesting, highlighting how in terms of 'game speed', there is not a disadvantage between using any model. 

## Demo links
HF Spaces: https://huggingface.co/spaces/build-small-hackathon/silent-frontier

Demo video: https://youtu.be/UyvjY-OsLWA (for HuggingFace x Gradio Build Small hackathon)

Built and extended on by **mkhi238** for the Hugging Face × Gradio Build Small hackathon.

## Getting started

### Prerequisites
- Python 3.11
- API keys for the model providers you want to use (Groq, NVIDIA, Cohere, Mistral) and an OpenAI key if you want to run the evaluation harness.

### Setup
```bash
# clone the repo
git clone https://github.com/mkhi238/npc_agent.git
cd npc_agent

# create and activate a virtual environment
python3.11 -m venv env
source env/bin/activate        # on Windows: env\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### Environment variables
Create a `.env` file in the project root with the keys for the models you intend to run:
GROQ_API_KEY=your_key_here

NVIDIA_API_KEY=your_key_here

COHERE_API_KEY=your_key_here

MISTRAL_API_KEY=your_key_here

OPENAI_API_KEY=your_key_here   # only needed to run eval.py

### Running the game
```bash
# launch the Gradio interface
python app.py
```
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

To play from the command line instead:
```bash
python main.py
```

### Running the evaluation
```bash
python eval.py
```
This runs the multi-model comparison and saves the score and latency charts as PNGs.
