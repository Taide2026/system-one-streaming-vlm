"""Every question prompt of the configs. A config picks one with question: {prompt: <name>} (s1stream.config.load_config
fills state, text, options and positive from PROMPTS; keys given in the config win).

Our first prompts, retired on 2026-10-23 for v1 and v2:
yes_no  "Is something abnormal happening?"; score p(Yes).
c14  UCF-Crime's 13 anomaly classes + Normal by name; score 1 - p(Normal). Arrest is "Police arrest" (for Gemma 4 E4B
    "Arrest" and "Arson" share their first token) and RoadAccidents is "Road accident".
yes_no_hint  yes_no with examples, the question of the jev_omni_ucf20 run.

The variants that follow the prompts of online VAD methods:

v1  MoniTor (NeurIPS 2025): its anomaly prior, the 13 activities and their definitions verbatim from its released
    context prompt (github.com/YsTvT/MoniTor, slurm/ucf_crime/04_summary.sh, "Recommended"): disorderly conduct and
    theft as there, no road accident. Its 0-1 rating becomes Yes / No; score p(Yes).
v2  ESOM (arXiv 2026): its normalized definition, a markdown table with anonymous options, what counts, what does not
    count and the event boundary, plus None; score 1 - p(None). Counts: MoniTor's definitions (H: Wikipedia, Traffic
    collision). Does not count and the boundary are ours: ESOM's own tables are not released. V2_CLASSES maps the
    options to the UCF-Crime classes. None comes last: decision runtimes list the options under their own codes A, B,
    ... in order, so with None last each letter's code is the letter itself (A: A, ..., M: M, N: None).
v3  ReactVAU (ECCV 2026): its fast model's question, without "Analyze the 2x2 video grid." (frames here, no grid);
    score p(Yes).
v1_xd, v2_xd  v1 and v2 for XD-Violence: its six classes, defined as in MoniTor's XD-Violence context prompt
    (slurm/xd_violence/04_summary.sh, the one with definitions), and STATE_XD (movies, sports, news and CCTV, not
    only surveillance). V2_XD_CLASSES maps the options to the XD-Violence classes (datasets/xd_violence.LABELS).
"""
from __future__ import annotations

STATE = "Surveillance camera footage, 2 seconds."

YES_NO_TEXT = "Is something abnormal happening?"
YES_NO_HINT_TEXT = "Is something abnormal happening, such as a crime, an accident, a fight, a fire or an explosion?"
C14_TEXT = "What is happening?"
C14_OPTIONS = ["Normal", "Abuse", "Police arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
               "Road accident", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

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

STATE_XD = "Video footage, 2 seconds."

V1_XD_TEXT = (
    "Would you rate this scene as a scene involving suspicious or potential criminal activities, such as abuse "
    "(inflicting improper harm or misconduct on others), car accident (collision events involving motor vehicles on "
    "roads that may cause property damage or personal injury), explosion (phenomenon where substances release energy "
    "violently due to chemical reactions or physical processes), fighting (physical conflicts between two or more "
    "parties due to disputes, revenge, or other reasons), riot (a group of people protesting or resisting against the "
    "government or other organizations in a violent manner), or shooting (act of firing with firearms, possibly "
    "involving murder or assault), rather than a normal scene?"
)

V2_XD_CLASSES = {"A": "Abuse", "B": "Car accident", "C": "Explosion", "D": "Fighting", "E": "Riot", "F": "Shooting"}
# (option, counts, does not count), in V2_XD_CLASSES order.
V2_XD_ROWS = (
    ("A", "inflicting improper harm or misconduct on others", "play, care, or both sides hitting (D)"),
    ("B", "collision events involving motor vehicles on roads that may cause property damage or personal injury",
     "heavy traffic, a near miss without contact"),
    ("C", "phenomenon where substances release energy violently due to chemical reactions or physical processes",
     "smoke or fire without a blast"),
    ("D", "physical conflicts between two or more parties due to disputes, revenge, or other reasons",
     "sports, play fighting"),
    ("E", "a group of people protesting or resisting against the government or other organizations in a violent manner",
     "peaceful protests, marches, cheering crowds"),
    ("F", "act of firing with firearms, possibly involving murder or assault", "holding or showing a gun without firing"),
)
V2_XD_TEXT = "\n".join([
    "Which defined event happens in these 2 seconds? Answer with its option, or None if no defined event happens.",
    "An event counts from its first act (the first blow, shot, blast or contact) until it stops.",
    "",
    "| Option | Counts | Does not count |",
    "|---|---|---|",
    *(f"| {option} | {counts} | {excluded} |" for option, counts, excluded in V2_XD_ROWS),
    "| None | no defined event: people, vehicles and scenes as usual | |",
])


# UCF101 (2026-10-28), the generalization test: the 101 action classes of classInd.txt as the options, in its order;
# the "positive" is a formality (the runner wants one), accuracy is the argmax (scripts/ucf101-accuracy.py).
UCF101_CLASSES = (
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "Hammering",
    "HammerThrow",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpingJack",
    "JumpRope",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
)
STATE_UCF101 = "A short video clip of one human action."
UCF101_TEXT = "Which action is shown in the clip?"

PROMPTS: dict[str, dict] = {
    "yes_no": {"state": STATE, "text": YES_NO_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
    "yes_no_hint": {"state": STATE, "text": YES_NO_HINT_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
    "c14": {"state": STATE, "text": C14_TEXT, "options": C14_OPTIONS, "positive": C14_OPTIONS[1:]},
    "v1": {"state": STATE, "text": V1_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
    "v2": {"state": STATE, "text": V2_TEXT, "options": [*V2_CLASSES, "None"], "positive": list(V2_CLASSES)},
    "v3": {"state": STATE, "text": V3_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
    "v1_xd": {"state": STATE_XD, "text": V1_XD_TEXT, "options": ["Yes", "No"], "positive": "Yes"},
    "ucf101": {"state": STATE_UCF101, "text": UCF101_TEXT, "options": list(UCF101_CLASSES), "positive": UCF101_CLASSES[0]},
    "v2_xd": {"state": STATE_XD, "text": V2_XD_TEXT, "options": [*V2_XD_CLASSES, "None"],
              "positive": list(V2_XD_CLASSES)},
}
