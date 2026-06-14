# 🛰️ Silent Frontier

An NPC dialogue reasoning agent. Interrogate a chain of seven characters across a sci-fi mystery, earn each one's clue, and follow the trail from Commander Orion to the Warden.

## 🧠 Built small

The default text model is **NVIDIA Nemotron Mini 4B**, a 4B model fine-tuned for roleplay and game NPCs, served through NVIDIA's free hosted endpoint. A selector also offers **Groq Llama 3.1 8B** as an alternate. Every model used is under the 32B cap.

### ⚙️ How it's built

DSPy reasoning agent with best-of-N sampling, FAISS retrieval over a hand-authored lore graph, a neurosymbolic constraint checker, per-NPC SQLite memory, Orpheus voice, and a custom Gradio dashboard with a node map, case file, and live reasoning trace.
---

Built by **mkhi238** for the Hugging Face × Gradio Build Small hackathon.