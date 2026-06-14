---
title: Silent Frontier
emoji: 🛰️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.17.3
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: A reasoning agent that role-plays seven NPCs.
---

# 🛰️ Silent Frontier

An NPC dialogue reasoning agent. Interrogate a chain of seven characters across a sci-fi mystery, earn each one's clue, and follow the trail from Commander Orion to the Warden.

## 🧠 Built small

The default text model is **NVIDIA Nemotron Mini 4B**, a 4B model fine-tuned for roleplay and game NPCs, served through NVIDIA's free hosted endpoint. A selector also offers **Groq Llama 3.1 8B** as an alternate. Every model used is under the 32B cap.

## ⚙️ How it's built

DSPy reasoning agent with best-of-N sampling, FAISS retrieval over a hand-authored lore graph, a neurosymbolic constraint checker, per-NPC SQLite memory, Orpheus voice, and a custom Gradio dashboard with a node map, case file, and live reasoning trace.

## 🎬 Demo

**Demo video:** DEMO_VIDEO_LINK_HERE

**Social post:** SOCIAL_POST_LINK_HERE

---

Built by **mkhi238** for the Hugging Face × Gradio Build Small hackathon.