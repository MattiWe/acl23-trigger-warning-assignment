from dataclasses import dataclass


@dataclass
class Ao3Module:
    module_type: str
    heading: str
    content: str


def get_module_attrs(module_wrapper):
    heading_elem = module_wrapper.query_selector("h3.heading")
    content_elem = module_wrapper.query_selector("blockquote.userstuff")
    mod_type = '_'.join([c for c in module_wrapper.class_list if not c == 'module'])
    heading = heading_elem.text if heading_elem is not None else None
    content = content_elem.html if content_elem is not None else None

    return mod_type, heading, content
