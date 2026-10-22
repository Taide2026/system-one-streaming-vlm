"""Prompt variants that follow the prompts of online VAD methods. A config picks one with question: {prompt: <name>}
(s1stream.config.load_config fills state, text, options and positive from PROMPTS; keys given in the config win).

v1  MoniTor (NeurIPS 2025): its anomaly prior, the 13 activities and their definitions verbatim from its released
    context prompt (github.com/YsTvT/MoniTor, slurm/ucf_crime/04_summary.sh, "Recommended"): disorderly conduct and
    theft as there, no road accident. Its 0-1 rating becomes Yes / No; score p(Yes).
v2  ESOM (arXiv 2026): its normalized definition, a markdown table with anonymous options, what counts, what does not
    count and the event boundary, plus None; score 1 - p(None). Counts: MoniTor's definitions (H: Wikipedia, Traffic
    collision). Does not count and the boundary are ours: ESOM's own tables are not released. V2_CLASSES maps the
    options to the UCF-Crime classes.
v3  ReactVAU (ECCV 2026): its fast model's question, without "Analyze the 2x2 video grid." (frames here, no grid);
    score p(Yes).
"""
from __future__ import annotations

STATE = "Surveillance camera footage, 2 seconds."

V1_TEXT = (
    "If you were a law enforcement agency, would you rate this scene as a scene involving suspicious activities such "
    "as abuse (intentionally harming or mistreating others), arrest (legally detaining someone), arson (deliberately "
    "setting fire), assault (physical attack on someone), burglary (illegally entering with the intent to commit a "
    "crime), disorderly conduct (disruptive or destructive behavior), explosion (violent release of energy), fighting "
    "(violent confrontation), robbery (unlawfully taking property), shooting (firing a gun), shoplifting (stealing "
    "from a retail store), theft (taking someone else's property without permission), or vandalism (deliberate "
    "destruction of property), rather than a standard scene?"
)

V2_CLASSES = {"A": "Abuse", "B": "Arrest", "C": "Arson", "D": "Assault", "E": "Burglary", "F": "Explosion",
              "G": "Fighting", "H": "RoadAccidents", "I": "Robbery", "J": "Shooting", "K": "Shoplifting",
              "L": "Stealing", "M": "Vandalism"}
# (option, counts, does not count), in V2_CLASSES order.
V2_ROWS = (
    ("A", "intentionally harming or mistreating others", "play, care, or both sides hitting (G)"),
    ("B", "legally detaining someone", "police only talking to, guiding or searching people"),
    ("C", "deliberately setting fire", "accidental fire, cooking, smoking"),
    ("D", "physical attack on someone", "both sides hitting (G), bumping in a crowd"),
    ("E", "illegally entering with the intent to commit a crime", "entering through an open door or with a key"),
    ("F", "violent release of energy", "smoke or fire without a blast (C)"),
    ("G", "violent confrontation", "sports, play fighting, attacking someone who does not fight back (D)"),
    ("H", "a vehicle colliding with another vehicle, a pedestrian, an animal or an obstruction",
     "heavy traffic, a near miss without contact"),
    ("I", "unlawfully taking property", "taking without force or threat (L), in a shop (K)"),
    ("J", "firing a gun", "holding or showing a gun without firing"),
    ("K", "stealing from a retail store", "paying customers carrying goods"),
    ("L", "taking someone else's property without permission",
     "with force or threat (I), in a shop (K), after breaking in (E)"),
    ("M", "deliberate destruction of property", "accidental damage, repair or demolition work"),
)
V2_TEXT = "\n".join([
    "Which defined event happens in these 2 seconds? Answer with its option, or None if no defined event happens.",
    "An event counts from its first act (the first blow, flame, blast, entry or contact) until it stops.",
    "",
    "| Option | Counts | Does not count |",
    "|---|---|---|",
    *(f"| {option} | {counts} | {excluded} |" for option, counts, excluded in V2_ROWS),
    "| None | no defined event: people walking, talking, working, driving or standing as usual | |",
])

V3_TEXT = "Is there any anomalous behavior, safety hazard, or event that disrupts the normal scene?"

PROMPTS: dict[str, dict] = {
    "v1": {"state": STATE, "text": V1_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
    "v2": {"state": STATE, "text": V2_TEXT, "options": ["None", *V2_CLASSES], "positive": list(V2_CLASSES)},
    "v3": {"state": STATE, "text": V3_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
}
