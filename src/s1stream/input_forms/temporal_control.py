"""New controlled frame sampling, using the existing reader and model interface.

2026-11-09 (paper v4, E4a / E5): rule-aware history modes on top of history4. They need the past scores of the
same detector, read from a stored R0 run (two-pass evaluation: the decision for window t uses only p_{t'}, t' < t):
  history4_oracle  analysis only: history frames inside a GT event, once that event has ended before the current
                   window starts, are replaced by the matching frame of the event's own pre-event history.
  history4_r1      evict after alarm: if p_{t-1} < tau, history frames from windows with p >= tau are replaced by the
                   centre frames of the most recent windows with p < tau_ref within the last L seconds
                   (fallback: the matching current frame).
  history4_r2      normal reference: the history is the centre frames of the 4 most recent windows with p < tau_ref
                   within the last L seconds, oldest first (fallback for missing slots: the matching current frame).
Params: scores (R0 scores.jsonl), tau, tau_ref, L (s), history_s.
"""
import json
import math
from s1stream.input_forms.frames import read_frames
from s1stream.registry import INPUT_FORMS
from s1stream.types import ModelInput
from streamvad.temporal import select, sample

_SCORES = {}


def _key(name):
    name = str(name)
    return name[:-4] if name.endswith(".mp4") else name


def _load(path):
    if path not in _SCORES:
        table = {}
        with open(path) as stream:
            for line in stream:
                if line.strip():
                    r = json.loads(line)
                    table[(_key(r["video"]), int(r["window"]))] = float(r["score"])
        _SCORES[path] = table
    return _SCORES[path]


@INPUT_FORMS.register("temporal_control")
class TemporalControl:
    RULES = ("history4_oracle", "history4_r1", "history4_r2")

    def __init__(self, mode="original", seed=0, history_s=6.0, scores=None, tau=0.5, tau_ref=None, L=30.0, W=2.0):
        self.mode, self.seed, self.history_s = mode, seed, history_s
        self.scores, self.tau, self.L, self.W = scores, float(tau), float(L), float(W)
        self.tau_ref = float(tau) if tau_ref is None else float(tau_ref)
        if mode in self.RULES[1:] and not scores:
            raise ValueError(f"{mode} needs scores= (the R0 run's scores.jsonl)")

    def _p(self, video, t):
        return _load(self.scores).get((video, t))

    def _bank(self, video, t):
        """Centre frames' windows t' < t, within L seconds, with p < tau_ref, most recent first."""
        out = []
        for tp in range(t - 1, max(-1, t - 1 - int(math.ceil(self.L / self.W))), -1):
            p = self._p(video, tp)
            if p is not None and p < self.tau_ref:
                out.append(tp)
        return out

    def _centre(self, tp, fps, n):
        return min(n - 1, max(0, int(round((tp * self.W + self.W / 2) * fps))))

    def build(self, item, window, question):
        if len(window.frame_indices) != 8:
            raise ValueError("temporal_control requires an eight-frame window without an additional context policy")
        if self.mode not in self.RULES:
            indices = select(self.mode, window.start_s, window.end_s, item.fps, item.n_frames,
                             item.name, window.index, self.seed, self.history_s)
            return ModelInput(images=read_frames(item.path, indices))
        fps, n, s, t, video = item.fps, item.n_frames, window.start_s, int(window.index), _key(item.name)
        cur = list(sample(s, window.end_s, 4, fps, n))
        if s == 0:
            return ModelInput(images=read_frames(item.path, cur + cur))
        hist = list(sample(max(0.0, s - self.history_s), s, 4, fps, n))
        if self.mode == "history4_oracle":
            for k, f in enumerate(hist):
                for a, b in item.events:
                    if a <= f <= b and s >= (b + 1) / fps:
                        if a == 0:
                            hist[k] = cur[k]
                        else:
                            pre = sample(max(0.0, a / fps - self.history_s), a / fps, 4, fps, n)
                            hist[k] = pre[k]
                        break
        elif self.mode == "history4_r1":
            last = self._p(video, t - 1)
            if last is not None and last < self.tau:
                bank = [self._centre(tp, fps, n) for tp in self._bank(video, t)]
                bi = 0
                for k, f in enumerate(hist):
                    p = self._p(video, int(f / fps // self.W))
                    if p is not None and p >= self.tau:
                        if bi < len(bank):
                            hist[k] = bank[bi]; bi += 1
                        else:
                            hist[k] = cur[k]
                hist = sorted(hist)
        else:  # history4_r2
            bank = [self._centre(tp, fps, n) for tp in self._bank(video, t)[:4]]
            bank = sorted(bank)
            hist = [cur[k] for k in range(4 - len(bank))] + bank
        return ModelInput(images=read_frames(item.path, hist + cur))
