"""Import every plugin module once, so each registers its name. Add a new plugin module here."""
import s1stream.backbones.jev_omni  # noqa: F401
import s1stream.datasets.ucf_crime  # noqa: F401
import s1stream.heads.option_logits  # noqa: F401
import s1stream.heads.option_prob  # noqa: F401
import s1stream.input_forms.frames  # noqa: F401
import s1stream.rules.threshold  # noqa: F401
