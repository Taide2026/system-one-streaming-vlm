"""Import every plugin module once, so each registers its name. Add a new plugin module here."""
import s1stream.backbones.clef_flash  # noqa: F401
import s1stream.backbones.d3  # noqa: F401
import s1stream.backbones.qwen_first_token  # noqa: F401
import s1stream.backbones.shieldstral  # noqa: F401  (2026-11-13)
import s1stream.datasets.ucf101  # noqa: F401
import s1stream.datasets.ubnormal  # noqa: F401
import s1stream.datasets.complexvad  # noqa: F401  (2026-11-13)
import s1stream.datasets.msad  # noqa: F401  (2026-11-13)
import s1stream.datasets.ucf_crime  # noqa: F401
import s1stream.datasets.xd_violence  # noqa: F401
import s1stream.heads.option_logits  # noqa: F401
import s1stream.input_forms.diff  # noqa: F401  (2026-11-07, roadmap B2)
import s1stream.input_forms.foveated  # noqa: F401
import s1stream.input_forms.frames  # noqa: F401
import s1stream.input_forms.keycrop  # noqa: F401  (2026-11-13)
import s1stream.input_forms.temporal_control  # noqa: F401  (2026-11-08)
import s1stream.input_forms.upsample  # noqa: F401  (2026-11-11)
import s1stream.rules.threshold  # noqa: F401
