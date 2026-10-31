"""Folders of the in-video localization analyses (see ../README.md).

ENCODER_RUN holds the encoder probing run these analyses start from: its config.yaml (the 117 short test videos)
and its fit-<feature folder>/ probe fits. Each topic folder holds the outputs and logs of one question.
"""

from pathlib import Path

LOC = Path(__file__).resolve().parents[1]
ENCODER_RUN = LOC.parent / "encoder_probe_ucf_short_v1_f8" / "20261028-five-encoders"
FRAME_SPLIT = LOC / "01-frame-auc-split"
GENERALIZATION = LOC / "02-generalization"
SCENE_REMOVAL = LOC / "03-scene-removal"
LOSS_CHANGES = LOC / "04-loss-changes"
EVENT_PHASE = LOC / "05-event-phase"
MOTION = LOC / "06-motion-tempdiff"
FUSION = LOC / "07-fusion"
BIAS_EXPERT = LOC / "08-bias-expert"
