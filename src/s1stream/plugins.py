"""Import every plugin module once, so each registers its name. Add a new plugin module here."""
import s1stream.backbones.clef_flash  # noqa: F401
import s1stream.backbones.d1  # noqa: F401
import s1stream.backbones.d3  # noqa: F401
import s1stream.backbones.gemma_generate  # noqa: F401
import s1stream.backbones.h2o_lightning  # noqa: F401
import s1stream.backbones.imajev  # noqa: F401
import s1stream.backbones.jev_omni  # noqa: F401
import s1stream.backbones.qwen_first_token  # noqa: F401
import s1stream.backbones.startlux  # noqa: F401
import s1stream.datasets.ucf_crime  # noqa: F401
import s1stream.datasets.ucf_crime_train  # noqa: F401
import s1stream.datasets.xd_violence  # noqa: F401
import s1stream.heads.generated_answer  # noqa: F401
import s1stream.heads.option_logits  # noqa: F401
import s1stream.heads.option_prob  # noqa: F401
import s1stream.input_forms.frames  # noqa: F401
import s1stream.rules.threshold  # noqa: F401
