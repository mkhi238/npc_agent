import gradio as gr
from retriever import FAISSRetriever
from constraints import ConstraintChecker
from agent import NPCAgent, check_duplicates, candidate_generator
from character import NPCNode
from config import lore_data, INDEX_PATH, MAX_ATTEMPTS, CLUE_THRESHOLD, MAX_MESSAGES_BEFORE_CLUE, configure_lm

configure_lm()
agent = NPCAgent()
retriever = FAISSRetriever(index_dir=INDEX_PATH)
checker = ConstraintChecker(lore_data)

# instantiate all NPCs
orion  = NPCNode.from_lore(lore_data, "Commander Orion")
sorel  = NPCNode.from_lore(lore_data, "Admiral Sorel")
lira   = NPCNode.from_lore(lore_data, "Lira Dawn")
shade7 = NPCNode.from_lore(lore_data, "Shade-7")
ren    = NPCNode.from_lore(lore_data, "Ren Ashford")
elandra= NPCNode.from_lore(lore_data, "Sage Elandra")
warden = NPCNode.from_lore(lore_data, "The Warden")

for npc in [orion, sorel, lira, shade7, ren, elandra, warden]:
    npc.clear_memory()

orion.next_npc  = sorel
sorel.next_npc  = lira
lira.next_npc   = shade7
shade7.next_npc = ren
ren.next_npc    = elandra
elandra.next_npc= warden
warden.next_npc = None

orion.clue_topic   = "Silent Frontier Expedition"
sorel.clue_topic   = "why the expedition was authorized"
lira.clue_topic    = "navigational data route"
shade7.clue_topic  = "what the expedition found"
ren.clue_topic     = "Vorne's Log"
elandra.clue_topic = "Echo Prism memory fragment"
warden.clue_topic  = "Resonance Point structure"

NPC_MAP = {
    "Commander Orion": orion,
    "Admiral Sorel":   sorel,
    "Lira Dawn":       lira,
    "Shade-7":         shade7,
    "Ren Ashford":     ren,
    "Sage Elandra":    elandra,
    "The Warden":      warden,
}

# node positions on the SVG (cx, cy) in a 600x500 canvas
NODE_POSITIONS = {
    "Commander Orion": (300, 60),
    "Admiral Sorel":   (480, 150),
    "Lira Dawn":       (480, 280),
    "Shade-7":         (380, 390),
    "Ren Ashford":     (180, 390),
    "Sage Elandra":    (120, 270),
    "The Warden":      (120, 150),
}

CHAIN_ORDER = [
    ("Commander Orion", "Admiral Sorel"),
    ("Admiral Sorel",   "Lira Dawn"),
    ("Lira Dawn",       "Shade-7"),
    ("Shade-7",         "Ren Ashford"),
    ("Ren Ashford",     "Sage Elandra"),
    ("Sage Elandra",    "The Warden"),
]
ENTRY_NPC = CHAIN_ORDER[0][0]   # "Commander Orion"

# track clue state per NPC
clue_unlocked_state = {name: False for name in NPC_MAP}

def build_svg(active_npc, visited):
    lines = []
    # connection lines
    for src, dst in CHAIN_ORDER:
        if not clue_unlocked_state.get(src, False):
            continue
        x1, y1 = NODE_POSITIONS[src]
        x2, y2 = NODE_POSITIONS[dst]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#4af0c4" stroke-width="2" stroke-dasharray="6,4" opacity="0.7"/>'
        )

    # nodes
    targets = set()
    for name, unlocked in clue_unlocked_state.items():
        if unlocked:
            nxt = NPC_MAP[name].next_npc
            if nxt is not None and not clue_unlocked_state.get(nxt.name, False):
                targets.add(nxt.name)
    for name, (cx, cy) in NODE_POSITIONS.items():
        is_active  = name == active_npc
        is_target  = name in targets
        is_visited = name in visited
        is_unlocked = clue_unlocked_state.get(name, False)

        if is_active:
            ring_color  = "#4af0c4"
            fill_color  = "#0d2e2e"
            text_color  = "#4af0c4"
            ring_width  = 3
            glow = f'<circle cx="{cx}" cy="{cy}" r="32" fill="none" stroke="#4af0c4" stroke-width="8" opacity="0.18"/>'
        elif is_target:
            ring_color  = "#f0c44a"
            fill_color  = "#2e2410"
            text_color  = "#f0c44a"
            ring_width  = 2.5
            glow = f'<circle cx="{cx}" cy="{cy}" r="32" fill="none" stroke="#f0c44a" stroke-width="8" opacity="0.18"/>'

        elif is_visited:
            ring_color  = "#2a6e6e"
            fill_color  = "#0a1f1f"
            text_color  = "#8ecfcf"
            ring_width  = 2
            glow = ""
        else:
            ring_color  = "#1a3a3a"
            fill_color  = "#080f0f"
            text_color  = "#3a6060"
            ring_width  = 1.5
            glow = ""

        clue_dot = ""
        if is_unlocked:
            clue_dot = f'<circle cx="{cx+18}" cy="{cy-18}" r="5" fill="#f0c44a" stroke="#0d0d0d" stroke-width="1"/>'

        short = name.split()[0] if name != "The Warden" else "Warden"
        faction = (NPC_MAP[name].faction or "Unknown")[:12]

        lines.append(glow)
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="28" fill="{fill_color}" '
            f'stroke="{ring_color}" stroke-width="{ring_width}"/>'
        )
        lines.append(clue_dot)
        lines.append(
            f'<text x="{cx}" y="{cy+4}" text-anchor="middle" '
            f'font-family="monospace" font-size="10" fill="{text_color}" font-weight="bold">'
            f'{short}</text>'
        )
        lines.append(
            f'<text x="{cx}" y="{cy+44}" text-anchor="middle" '
            f'font-family="monospace" font-size="8" fill="{text_color}" opacity="0.7">'
            f'{faction}</text>'
        )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 500" width="100%" height="100%"
         style="background:#05080d; border-radius:8px;">
      <!-- title -->
      <text x="300" y="490" text-anchor="middle" font-family="monospace"
            font-size="10" fill="#1a4a4a" letter-spacing="4">
        SILENT FRONTIER - INVESTIGATION NODE MAP
      </text>
      {''.join(lines)}
    </svg>
    """
    return svg


def run_conversation(player_input, history, active_npc_name, visited_str, reasoning_log):
    if not player_input.strip():
        return history, reasoning_log, visited_str, build_svg(active_npc_name, set(visited_str.split(",")))

    npc = NPC_MAP[active_npc_name]
    visited = set(v for v in visited_str.split(",") if v)
    visited.add(active_npc_name)

    game_state = npc.build_game_state(player_input)

    state_results  = retriever.retrieve(player_input, k=3)
    npc_results    = retriever.retrieve(npc.name, k=2)
    retrieved      = check_duplicates(state_results + npc_results)
    lore_context   = " ".join([retriever.flatten_entry(r) for r in retrieved])

    if npc.clue and not clue_unlocked_state[active_npc_name]:
        if active_npc_name == ENTRY_NPC:
            clue_unlocked_state[active_npc_name] = True
        else:
            sim = retriever.measure_clue_similarity(player_input, npc.clue_topic)
            if sim > CLUE_THRESHOLD or npc.message_count >= MAX_MESSAGES_BEFORE_CLUE:
                clue_unlocked_state[active_npc_name] = True

    clue_unlocked = clue_unlocked_state[active_npc_name]

    npc_secret = (
        f"Topic: {npc.clue_topic}. Secret: {npc.clue}"
        if clue_unlocked
        else f"Steer conversation toward: {npc.clue_topic} without revealing details yet."
    )
    next_npc = npc.next_npc.name if (clue_unlocked and npc.next_npc) else ""

    best = None
    best_ruling = ""
    best_justification = ""
    best_flags = []

    for attempt in range(MAX_ATTEMPTS):
        results = agent(
            game_state=game_state,
            lore_context=lore_context,
            npc_name=npc.name,
            npc_personality=f"Speaking style: {npc.speaking_style}; Personality: {npc.personality}",
            npc_secret=npc_secret,
            next_npc=next_npc,
        )
        generator = candidate_generator(results, checker, npc.name)
        found_pass = False
        for candidate, ruling, justification, flags in generator:
            if ruling.strip().upper() == "PASS":
                best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
                found_pass = True
                break
            if best is None or len(flags) < len(best_flags):
                best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
        if found_pass:
            break

    npc.store_exchange(player_input, best.dialogue)

    history = history or []
    history.append({"role": "user", "content": player_input})
    history.append({"role": "assistant", "content": best.dialogue})

    turn_no = len(history) // 2
    clue_badge = "🟡 CLUE UNLOCKED" if clue_unlocked else "🔒 Investigating..."
    log_entry = (
        f"--- Turn {turn_no} | {active_npc_name} | {clue_badge} ---\n"
        f"RULING: {best_ruling}\n"
        f"JUSTIFICATION: {best_justification}\n"
        f"FLAGS: {best_flags if best_flags else 'None'}\n"
        f"REASONING: {best.reasoning}\n"
    )
    reasoning_log = (reasoning_log or "") + "\n" + log_entry

    new_svg = build_svg(active_npc_name, visited)
    return history, reasoning_log, ",".join(visited), new_svg


def switch_npc(npc_name, visited_str):
    visited = set(v for v in visited_str.split(",") if v)
    svg = build_svg(npc_name, visited)
    npc = NPC_MAP[npc_name]
    info = f"**{npc.name}** | {npc.faction or 'Unknown'}\n\n_{npc.personality}_"
    return [], "", npc_name, info, svg


CSS = """
body, .gradio-container { background: #05080d !important; color: #c0d8d8 !important; }
.gr-button { background: #0d2e2e !important; color: #4af0c4 !important;
             border: 1px solid #1a5a5a !important; font-family: monospace !important; }
.gr-button:hover { background: #1a4a4a !important; }
.gr-textbox textarea, .gr-textbox input {
    background: #080f0f !important; color: #c0d8d8 !important;
    border: 1px solid #1a3a3a !important; font-family: monospace !important; }
.gr-chatbot { background: #080f0f !important; border: 1px solid #1a3a3a !important; }
.gr-dropdown select { background: #080f0f !important; color: #4af0c4 !important;
                      border: 1px solid #1a3a3a !important; font-family: monospace !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Silent Frontier - NPC Investigation") as demo:
    # state
    active_npc_state  = gr.State("Commander Orion")
    visited_state     = gr.State("")

    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 8px 0;">
      <span style="font-family:monospace; font-size:22px; letter-spacing:6px; color:#4af0c4;">
        SILENT FRONTIER
      </span>
      <br>
      <span style="font-family:monospace; font-size:11px; color:#2a6e6e; letter-spacing:3px;">
        NPC DIALOGUE REASONING AGENT - INVESTIGATION INTERFACE
      </span>
    </div>
    """)

    with gr.Row():
        # LEFT: node map
        with gr.Column(scale=4):
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;'>INVESTIGATION MAP</p>")
            svg_display = gr.HTML(build_svg("Commander Orion", set()))

            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;margin-top:12px;'>SELECT CONTACT</p>")
            npc_dropdown = gr.Dropdown(
                choices=list(NPC_MAP.keys()),
                value="Commander Orion",
                label="",
                show_label=False,
            )
            npc_info = gr.Markdown(
                f"**Commander Orion** | Vanguard Coalition\n\n_formal, burdened by command, deeply loyal_"
            )
            switch_btn = gr.Button("OPEN CHANNEL", size="sm")

        # RIGHT: chat
        with gr.Column(scale=6):
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;'>TRANSMISSION LOG</p>")
            chatbot = gr.Chatbot(
                label="",
                height=380,
            )
            with gr.Row():
                player_input = gr.Textbox(
                    placeholder="Transmit message...",
                    label="",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("TRANSMIT", scale=1)

            with gr.Accordion("REASONING TRACE + CONSTRAINT LOG", open=False):
                reasoning_display = gr.Textbox(
                    label="",
                    show_label=False,
                    lines=10,
                    interactive=False,
                    placeholder="Constraint checker and reasoning trace will appear here...",
                )

    # events
    switch_btn.click(
        fn=switch_npc,
        inputs=[npc_dropdown, visited_state],
        outputs=[chatbot, reasoning_display, active_npc_state, npc_info, svg_display],
    )

    send_btn.click(
        fn=run_conversation,
        inputs=[player_input, chatbot, active_npc_state, visited_state, reasoning_display],
        outputs=[chatbot, reasoning_display, visited_state, svg_display],
    ).then(lambda: "", outputs=player_input)

    player_input.submit(
        fn=run_conversation,
        inputs=[player_input, chatbot, active_npc_state, visited_state, reasoning_display],
        outputs=[chatbot, reasoning_display, visited_state, svg_display],
    ).then(lambda: "", outputs=player_input)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)