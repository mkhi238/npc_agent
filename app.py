import gradio as gr
from retriever import FAISSRetriever
from constraints import ConstraintChecker
from agent import NPCAgent, check_duplicates, candidate_generator
from character import NPCNode
from config import lore_data, INDEX_PATH, MAX_ATTEMPTS, CLUE_THRESHOLD, MAX_MESSAGES_BEFORE_CLUE, configure_lm, MODEL_REGISTRY, DEFAULT_MODEL
from tts import synthesize
import os

configure_lm()
agent = NPCAgent()
retriever = FAISSRetriever(index_dir=INDEX_PATH)
checker = ConstraintChecker(lore_data)

# instantiate all NPCs
orion   = NPCNode.from_lore(lore_data, "Commander Orion")
sorel   = NPCNode.from_lore(lore_data, "Admiral Sorel")
lira    = NPCNode.from_lore(lore_data, "Lira Dawn")
shade7  = NPCNode.from_lore(lore_data, "Shade-7")
ren     = NPCNode.from_lore(lore_data, "Ren Ashford")
elandra = NPCNode.from_lore(lore_data, "Sage Elandra")
warden  = NPCNode.from_lore(lore_data, "The Warden")

# chain progression
orion.next_npc   = sorel
sorel.next_npc   = lira
lira.next_npc    = shade7
shade7.next_npc  = ren
ren.next_npc     = elandra
elandra.next_npc = warden
warden.next_npc  = None

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

# Orpheus voice per NPC (Groq orpheus-v1-english voices).
# Male: troy, austin, daniel.  Female: autumn, diana, hannah.
VOICE_MAP = {
    "Commander Orion": "troy",
    "Admiral Sorel":   "diana",
    "Lira Dawn":       "hannah",
    "Shade-7":         "daniel",
    "Ren Ashford":     "austin",
    "Sage Elandra":    "autumn",
    "The Warden":      "diana",
}

# Character portrait per NPC (files live in ./assets, anchored to this file's dir).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MAP = {
    "Commander Orion": os.path.join(_APP_DIR, "assets", "orion.png"),
    "Admiral Sorel":   os.path.join(_APP_DIR, "assets", "sorel.png"),
    "Lira Dawn":       os.path.join(_APP_DIR, "assets", "lira.png"),
    "Shade-7":         os.path.join(_APP_DIR, "assets", "shade7.png"),
    "Ren Ashford":     os.path.join(_APP_DIR, "assets", "ren.png"),
    "Sage Elandra":    os.path.join(_APP_DIR, "assets", "elandra.png"),
    "The Warden":      os.path.join(_APP_DIR, "assets", "warden.png"),
}

# Short "what you learned" label shown in progress table and case file once unlocked.
CLUE_LABEL = {
    "Commander Orion": "The expedition went to Resonance Point and all but one were lost",
    "Admiral Sorel":   "Sorel authorized it after an anomalous reading from Resonance Point",
    "Lira Dawn":       "She sold the expedition its navigation route",
    "Shade-7":         "The Eclipse Syndicate intercepted Vorne's final transmission",
    "Ren Ashford":     "Ren deserted and kept Vorne's Log",
    "Sage Elandra":    "The Echo Prism holds a memory of Resonance Point",
    "The Warden":      "The structure at Resonance Point predates the Coalition",
}

# Two guiding hints per NPC. Hint 1 is a soft nudge, hint 2 is the direct ask.
HINTS = {
    "Commander Orion": ["He leads the Vanguard Coalition. Press him on what they were exploring out there.",
                        "Ask him directly about the Silent Frontier Expedition."],
    "Admiral Sorel":   ["She answers to the Solar Accord. Press on who gave the order and why.",
                        "Ask why the expedition was authorized."],
    "Lira Dawn":       ["She is Eclipse Syndicate. Ask what path the expedition followed.",
                        "Ask about the navigational data and the route they took."],
    "Shade-7":         ["Shade-7 keeps Eclipse Syndicate records. Ask what was discovered out there.",
                        "Ask what the expedition actually found."],
    "Ren Ashford":     ["Ren of the Iron Covenant knows of a recovered log. Ask whose it was.",
                        "Ask about Vorne's Log."],
    "Sage Elandra":    ["Elandra of the Celestial Order reads relics. Ask what the fragment holds.",
                        "Ask about the Echo Prism memory fragment."],
    "The Warden":      ["The Warden guards the final truth. Ask what the Resonance Point is.",
                        "Ask about the Resonance Point structure."],
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
ENTRY_NPC = CHAIN_ORDER[0][0]    # "Commander Orion"
FINAL_NPC = CHAIN_ORDER[-1][1]   # "The Warden"


def fresh_state():
    """A clean per-game progress dict, all NPCs locked / undelivered."""
    return {name: False for name in NPC_MAP}


def chain_names():
    names, cur = [], NPC_MAP[ENTRY_NPC]
    while cur is not None:
        names.append(cur.name)
        cur = cur.next_npc
    return names


def build_progress(unlocked):
    rows = ""
    for name in chain_names():
        npc = NPC_MAP[name]
        if unlocked.get(name, False):
            clue = CLUE_LABEL.get(name, npc.clue_topic)
            clue_cell = f"<td style='padding:4px 6px;color:#c0d8d8;'>{clue}</td>"
            name_color = "#4af0c4"
            nxt = npc.next_npc
            if nxt is None:
                next_cell = "<td style='padding:4px 6px;color:#f0c44a;'>Final</td>"
            elif unlocked.get(nxt.name, False):
                next_cell = f"<td style='padding:4px 6px;color:#f0c44a;'>{nxt.name}</td>"
            else:
                next_cell = "<td style='padding:4px 6px;color:#3a6060;'>???</td>"
        else:
            clue_cell = "<td style='padding:4px 6px;color:#3a6060;'>???</td>"
            next_cell = "<td style='padding:4px 6px;color:#3a6060;'>???</td>"
            name_color = "#5a7a7a"
        rows += (
            "<tr>"
            f"<td style='padding:4px 6px;color:{name_color};'>{name}</td>"
            f"{clue_cell}{next_cell}"
            "</tr>"
        )
    return (
        "<div style='font-family:monospace;font-size:10px;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<tr style='color:#2a6e6e;border-bottom:1px solid #1a3a3a;'>"
        "<th style='text-align:left;padding:4px 6px;'>CONTACT</th>"
        "<th style='text-align:left;padding:4px 6px;'>CLUE</th>"
        "<th style='text-align:left;padding:4px 6px;'>POINTS TO</th>"
        "</tr>"
        f"{rows}"
        "</table></div>"
    )


def build_casefile(unlocked):
    items = ""
    for name in chain_names():
        if unlocked.get(name, False):
            label = CLUE_LABEL.get(name, NPC_MAP[name].clue_topic)
            items += (
                "<div style='margin:6px 0;padding:8px 10px;border-left:2px solid #4af0c4;"
                "background:#0a1414;border-radius:0 4px 4px 0;'>"
                f"<span style='color:#4af0c4;font-weight:bold;'>{name}</span><br>"
                f"<span style='color:#c0d8d8;'>{label}</span></div>"
            )
    if not items:
        items = ("<div style='color:#3a6060;padding:8px 10px;'>"
                 "No clues recovered yet. Begin with Commander Orion.</div>")
    return f"<div style='font-family:monospace;font-size:11px;'>{items}</div>"


def build_status(active_npc_name, unlocked, sim):
    """Small indicator box: last query match score + locked/unlocked for active NPC."""
    if unlocked.get(active_npc_name, False):
        status = "<span style='color:#4af0c4;font-weight:bold;'>CLUE UNLOCKED</span>"
    else:
        status = "<span style='color:#f0c44a;font-weight:bold;'>LOCKED</span>"
    score = f"{sim:.2f}" if sim is not None else "--"
    return (
        "<div style='font-family:monospace;font-size:11px;color:#c0d8d8;"
        "border:1px solid #1a3a3a;border-radius:6px;padding:8px 12px;background:#080f0f;'>"
        f"QUERY MATCH: <span style='color:#8ecfcf;'>{score}</span>"
        "&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"STATUS: {status}"
        "</div>"
    )


def build_svg(active_npc, visited, unlocked):
    lines = []
    for src, dst in CHAIN_ORDER:
        if not unlocked.get(src, False):
            continue
        x1, y1 = NODE_POSITIONS[src]
        x2, y2 = NODE_POSITIONS[dst]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#4af0c4" stroke-width="2" stroke-dasharray="6,4" opacity="0.7"/>'
        )

    targets = set()
    for name, is_open in unlocked.items():
        if is_open:
            nxt = NPC_MAP[name].next_npc
            if nxt is not None and not unlocked.get(nxt.name, False):
                targets.add(nxt.name)

    for name, (cx, cy) in NODE_POSITIONS.items():
        is_active   = name == active_npc
        is_target   = name in targets
        is_visited  = name in visited
        is_unlocked = unlocked.get(name, False)

        if is_active:
            ring_color, fill_color, text_color, ring_width = "#4af0c4", "#0d2e2e", "#4af0c4", 3
            glow = f'<circle cx="{cx}" cy="{cy}" r="32" fill="none" stroke="#4af0c4" stroke-width="8" opacity="0.18"/>'
        elif is_target:
            ring_color, fill_color, text_color, ring_width = "#f0c44a", "#2e2410", "#f0c44a", 2.5
            glow = f'<circle cx="{cx}" cy="{cy}" r="32" fill="none" stroke="#f0c44a" stroke-width="8" opacity="0.18"/>'
        elif is_visited:
            ring_color, fill_color, text_color, ring_width = "#2a6e6e", "#0a1f1f", "#8ecfcf", 2
            glow = ""
        else:
            ring_color, fill_color, text_color, ring_width = "#1a3a3a", "#080f0f", "#3a6060", 1.5
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
            f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="{text_color}" font-weight="bold">{short}</text>'
        )
        lines.append(
            f'<text x="{cx}" y="{cy+44}" text-anchor="middle" font-family="monospace" '
            f'font-size="8" fill="{text_color}" opacity="0.7">{faction}</text>'
        )

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 500" width="100%" height="100%"
         style="background:#05080d; border-radius:8px;">
      <text x="300" y="490" text-anchor="middle" font-family="monospace"
            font-size="10" fill="#1a4a4a" letter-spacing="4">
        SILENT FRONTIER - INVESTIGATION NODE MAP
      </text>
      {''.join(lines)}
    </svg>
    """


def build_secret(npc, is_unlocked, is_delivered):
    """Build the npc_secret instruction and the next_npc handoff for this turn."""
    if is_unlocked and not is_delivered:
        if npc.next_npc:
            handoff = (
                f" After sharing this, tell the player in your own voice and in character "
                f"that {npc.next_npc.name} is the one they should speak to next."
            )
        else:
            handoff = (
                " This is the final revelation. The investigation ends here. "
                "Do not send the player to anyone else or invent further leads. "
                "Speak as if the trail is complete."
            )
        secret = f"Topic: {npc.clue_topic}. Share this openly with the player now: {npc.clue}.{handoff}"
        next_npc = npc.next_npc.name if npc.next_npc else ""
    elif is_unlocked and is_delivered:
        secret = (
            f"You have already told the player about {npc.clue_topic}, and they know it. "
            f"Continue talking in character. You may briefly reference it if asked, but do not "
            f"re-explain it at length and do not tell them who to speak to next again. "
            f"Just have a natural conversation."
        )
        next_npc = ""
    else:
        secret = (
            f"The player has not yet earned the full story of {npc.clue_topic}. "
            f"Answer in character in two or three sentences. Stay grounded in what you "
            f"actually know. Do not invent new names, ships, factions, or events, and do "
            f"not name other characters unless the player names them first. If the player "
            f"presses toward the topic, give a small real hint but hold the deepest "
            f"specifics. Keep it short and do not ramble."
        )
        next_npc = ""
    return secret, next_npc


def try_unlock(npc, active_npc_name, player_input, unlocked):
    """Run the unlock gate. Mutates `unlocked`. Returns the similarity score, or None."""
    if not npc.clue or unlocked[active_npc_name]:
        return None
    sim = retriever.measure_clue_similarity(player_input, npc.clue_topic)

    # The Warden stays locked until every other NPC is unlocked (soft gate).
    if active_npc_name == FINAL_NPC:
        others_done = all(unlocked[n] for n in NPC_MAP if n != FINAL_NPC)
        if not others_done:
            return sim  # show the score, but do not unlock yet

    if active_npc_name == ENTRY_NPC:
        if sim > 0.55 or npc.message_count >= 3:
            unlocked[active_npc_name] = True
    else:
        if sim > CLUE_THRESHOLD or npc.message_count >= MAX_MESSAGES_BEFORE_CLUE:
            unlocked[active_npc_name] = True
    return sim


def run_conversation(player_input, history, active_npc_name, visited_str,
                     reasoning_log, unlocked, delivered):
    if not player_input.strip():
        return (history, reasoning_log, visited_str,
                build_svg(active_npc_name, set(visited_str.split(",")), unlocked),
                None, build_progress(unlocked), build_casefile(unlocked),
                unlocked, delivered, build_status(active_npc_name, unlocked, None))

    npc = NPC_MAP[active_npc_name]

    visited = set(v for v in visited_str.split(",") if v)
    visited.add(active_npc_name)

    game_state   = npc.build_game_state(player_input)
    retrieved    = check_duplicates(retriever.retrieve(player_input, k=3) + retriever.retrieve(npc.name, k=2))
    lore_context = " ".join(retriever.flatten_entry(r) for r in retrieved)

    sim = try_unlock(npc, active_npc_name, player_input, unlocked)

    is_unlocked  = unlocked[active_npc_name]
    is_delivered = delivered[active_npc_name]
    npc_secret, next_npc = build_secret(npc, is_unlocked, is_delivered)
    if is_unlocked and not is_delivered:
        delivered[active_npc_name] = True

    best = None
    best_ruling = best_justification = ""
    best_flags = []

    for _ in range(MAX_ATTEMPTS):
        results = agent(
            game_state=game_state,
            lore_context=lore_context,
            npc_name=npc.name,
            npc_personality=f"Speaking style: {npc.speaking_style}; Personality: {npc.personality}",
            npc_secret=npc_secret,
            next_npc=next_npc,
        )
        found_pass = False
        for candidate, ruling, justification, flags in candidate_generator(results, checker, npc.name):
            if ruling.strip().upper() == "PASS":
                best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
                found_pass = True
                break
            if best is None or len(flags) < len(best_flags):
                best, best_ruling, best_justification, best_flags = candidate, ruling, justification, flags
        if found_pass:
            break

    npc.store_exchange(player_input, best.dialogue)
    audio_path = synthesize(best.dialogue, VOICE_MAP.get(active_npc_name, "troy"))

    history = history or []
    history.append({"role": "user", "content": player_input})
    history.append({"role": "assistant", "content": best.dialogue})

    turn_no = len(history) // 2
    clue_badge = "\U0001F7E1 CLUE UNLOCKED" if is_unlocked else "\U0001F512 Investigating..."
    log_entry = (
        f"--- Turn {turn_no} | {active_npc_name} | {clue_badge} ---\n"
        f"RULING: {best_ruling}\n"
        f"JUSTIFICATION: {best_justification}\n"
        f"FLAGS: {best_flags if best_flags else 'None'}\n"
        f"REASONING: {best.reasoning}\n"
    )
    reasoning_log = (reasoning_log or "") + "\n" + log_entry

    new_svg = build_svg(active_npc_name, visited, unlocked)
    return (history, reasoning_log, ",".join(visited), new_svg, audio_path,
            build_progress(unlocked), build_casefile(unlocked),
            unlocked, delivered, build_status(active_npc_name, unlocked, sim))


def switch_npc(npc_name, visited_str, unlocked):
    """Switch active contact. Updates view and hint text. Leaves hint open-state alone."""
    visited = set(v for v in visited_str.split(",") if v)
    svg = build_svg(npc_name, visited, unlocked)
    npc = NPC_MAP[npc_name]
    info = f"**{npc.name}** | {npc.faction or 'Unknown'}\n\n_{npc.personality}_"
    h1, h2 = HINTS.get(npc_name, ["", ""])
    portrait = IMAGE_MAP.get(npc_name)
    return ([], "", npc_name, info, svg, build_progress(unlocked), h1, h2,
            build_status(npc_name, unlocked, None), portrait)


def reset_game():
    """Wipe everything back to a fresh investigation. Used by load and the reset button."""
    for npc in NPC_MAP.values():
        npc.clear_memory()
    unlocked = fresh_state()
    delivered = fresh_state()
    npc = NPC_MAP[ENTRY_NPC]
    info = f"**{npc.name}** | {npc.faction or 'Unknown'}\n\n_{npc.personality}_"
    svg = build_svg(ENTRY_NPC, set(), unlocked)
    h1, h2 = HINTS[ENTRY_NPC]
    return (unlocked, delivered, [], "", ENTRY_NPC, "", info, svg,
            build_progress(unlocked), build_casefile(unlocked),
            h1, h2, build_status(ENTRY_NPC, unlocked, None), IMAGE_MAP.get(ENTRY_NPC))


def switch_model(choice):
    """Reconfigure DSPy's global LM to the chosen text model."""
    active = configure_lm(choice)
    return f"<div style='font-family:monospace;font-size:11px;color:#8ecfcf;'>Active text model: <span style='color:#4af0c4;'>{active}</span></div>"


CSS = """
body, .gradio-container { background: #05080d !important; color: #c0d8d8 !important; }
.gr-button { background: #0d2e2e !important; color: #4af0c4 !important;
             border: 1px solid #1a5a5a !important; font-family: monospace !important; }
.gr-button:hover { background: #1a4a4a !important; }
.gr-textbox textarea, .gr-textbox input {
    background: #080f0f !important; color: #c0d8d8 !important;
    border: 1px solid #1a3a3a !important; font-family: monospace !important; }
.gr-chatbot { background: #080f0f !important; border: 1px solid #1a3a3a !important; }
#npc_audio { max-width: 340px; }
#npc_portrait { max-width: 240px; margin-left: auto; margin-right: auto; border: 1px solid #1a3a3a; border-radius: 8px; }
#npc_portrait img { border-radius: 8px; }
#npc_portrait .icon-button-wrapper, #npc_portrait .toolbar { display: none !important; }
#npc_audio .waveform-container { display: none !important; }
#npc_audio .timestamps { display: none !important; }
footer { display: none !important; }
"""

INIT_UNLOCKED = fresh_state()

with gr.Blocks(title="Silent Frontier - NPC Investigation") as demo:
    # per-session state (resets on browser refresh)
    active_npc_state = gr.State("Commander Orion")
    visited_state    = gr.State("")
    unlocked_state   = gr.State(fresh_state())
    delivered_state  = gr.State(fresh_state())

    # title with a small reset button pinned to the right
    with gr.Row():
        gr.HTML("""
        <div style="padding: 16px 0 4px 0;">
          <span style="font-family:monospace; font-size:22px; letter-spacing:6px; color:#4af0c4;">
            SILENT FRONTIER
          </span><br>
          <span style="font-family:monospace; font-size:11px; color:#2a6e6e; letter-spacing:3px;">
            NPC DIALOGUE REASONING AGENT - INVESTIGATION INTERFACE
          </span>
        </div>
        """)
        reset_btn = gr.Button("RESET", size="sm", scale=0, min_width=90)

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(MODEL_REGISTRY.keys()), value=DEFAULT_MODEL,
            label="TEXT MODEL", scale=2,
        )
        model_status = gr.HTML(
            "<div style='font-family:monospace;font-size:11px;color:#8ecfcf;'>"
            f"Active text model: <span style='color:#4af0c4;'>{DEFAULT_MODEL}</span></div>"
        )

    with gr.Accordion("REASONING TRACE + CONSTRAINT LOG", open=False):
        reasoning_display = gr.Textbox(
            label="", show_label=False, lines=10, interactive=False,
            placeholder="Constraint checker and reasoning trace will appear here...",
        )

    with gr.Row():
        # LEFT: node map + active contact
        with gr.Column(scale=4):
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;'>INVESTIGATION MAP</p>")
            svg_display = gr.HTML(build_svg("Commander Orion", set(), INIT_UNLOCKED))

        # MIDDLE: chat + status + audio
        with gr.Column(scale=5):
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;'>TRANSMISSION LOG</p>")
            chatbot = gr.Chatbot(label="", height=380, buttons=["copy_all"])
            status_display = gr.HTML(build_status(ENTRY_NPC, INIT_UNLOCKED, None))
            audio_out = gr.Audio(label="", show_label=False, autoplay=False, buttons=[], elem_id="npc_audio")
            with gr.Row():
                player_input = gr.Textbox(placeholder="Transmit message...", label="", show_label=False, scale=5)
                send_btn = gr.Button("TRANSMIT", scale=1)

        # RIGHT: progress table + active contact
        with gr.Column(scale=3):
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;'>PROGRESS</p>")
            progress_display = gr.HTML(build_progress(INIT_UNLOCKED))
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;margin-top:12px;'>CONTACT</p>")
            npc_info = gr.Markdown(
                "**Commander Orion** | Vanguard Coalition\n\n_formal, burdened by command, deeply loyal_"
            )
            gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;margin-top:12px;text-align:center;'>NOW SPEAKING WITH</p>")
            portrait_display = gr.Image(value=IMAGE_MAP[ENTRY_NPC], show_label=False,
                                        height=260, interactive=False,
                                        container=False, elem_id="npc_portrait")

    # CASE FILE: stacked record of every clue earned so far
    gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;margin-top:12px;'>CASE FILE - RECOVERED CLUES</p>")
    casefile_display = gr.HTML(build_casefile(INIT_UNLOCKED))

    # HINTS: two manual accordions, default closed
    gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;margin-top:12px;'>HINTS</p>")
    with gr.Row():
        with gr.Accordion("HINT 1 (click to reveal)", open=False):
            hint1_md = gr.Markdown(HINTS[ENTRY_NPC][0])
        with gr.Accordion("HINT 2 (click to reveal)", open=False):
            hint2_md = gr.Markdown(HINTS[ENTRY_NPC][1])

    # SELECT CONTACT
    gr.HTML("<p style='font-family:monospace;font-size:10px;color:#2a6e6e;letter-spacing:2px;margin-top:12px;'>SELECT CONTACT</p>")
    with gr.Row():
        npc_buttons = {name: gr.Button(name, size="sm") for name in NPC_MAP}

    # ---- events ----
    convo_inputs = [player_input, chatbot, active_npc_state, visited_state, reasoning_display,
                    unlocked_state, delivered_state]
    convo_outputs = [chatbot, reasoning_display, visited_state, svg_display, audio_out,
                     progress_display, casefile_display, unlocked_state, delivered_state,
                     status_display]

    for trigger in (send_btn.click, player_input.submit):
        trigger(fn=run_conversation, inputs=convo_inputs, outputs=convo_outputs
                ).then(lambda: "", outputs=player_input)

    for name, btn in npc_buttons.items():
        btn.click(
            fn=lambda v, u, n=name: switch_npc(n, v, u),
            inputs=[visited_state, unlocked_state],
            outputs=[chatbot, reasoning_display, active_npc_state, npc_info, svg_display,
                     progress_display, hint1_md, hint2_md, status_display, portrait_display],
        )

    reset_outputs = [unlocked_state, delivered_state, chatbot, reasoning_display,
                     active_npc_state, visited_state, npc_info, svg_display,
                     progress_display, casefile_display, hint1_md, hint2_md, status_display,
                     portrait_display]
    reset_btn.click(fn=reset_game, inputs=None, outputs=reset_outputs)
    demo.load(fn=reset_game, inputs=None, outputs=reset_outputs)

    model_selector.change(fn=switch_model, inputs=model_selector, outputs=model_status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS,
                allowed_paths=[os.path.join(_APP_DIR, "assets")])