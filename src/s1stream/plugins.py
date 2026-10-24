"""Import every plugin module once, so each registers its name. Add a new plugin module here."""
import s1stream.backbones.clef_flash  # noqa: F401
import s1stream.backbones.d3  # noqa: F401
import s1stream.backbones.qwen_first_token  # noqa: F401
import s1stream.datasets.ucf_crime  # noqa: F401
import s1stream.datasets.xd_violence  # noqa: F401
import s1stream.heads.option_logits  # noqa: F401
import s1stream.input_forms.frames  # noqa: F401
import s1stream.rules.threshold  # noqa: F401
