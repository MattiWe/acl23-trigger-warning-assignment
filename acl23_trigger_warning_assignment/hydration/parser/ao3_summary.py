from dataclasses import dataclass

from .ao3_module import Ao3Module, get_module_attrs


@dataclass
class Ao3Summary(Ao3Module):
    pass


def process_summary(summary_wrapper):
    mod_type, heading, content = get_module_attrs(summary_wrapper)
    assert mod_type == "summary"

    return Ao3Summary(mod_type, heading, content)
