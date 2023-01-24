import re
from dataclasses import dataclass
from .ao3_module import Ao3Module, get_module_attrs


@dataclass
class Ao3Notes(Ao3Module):
    associations: str
    jumps: list[str]


def get_work_id(s):
    m = re.match(r'/works/(\d+)', s)
    if m is not None:
        return m.group(1)


def get_user(s):
    m = re.match(r'(/users/(.*)/)gifts', s)
    if m is not None:
        return m.group(2), m.group(1)


def get_association_type(association_elem):
    m = re.match(r'Translation into (.*?) available:\s*(.*)\s*by\s*(.*?)\s*', association_elem.text.strip())
    if m is not None:
        return 'translation_into'
    m = re.match(r'A translation of\s*(.*?)\s*by\s*(.*?)\.\s*', association_elem.text.strip())
    if m is not None:
        return 'translated_from'
    m = re.match(r'For\s*(.*)\.', association_elem.text.strip())
    if m is not None:
        return 'gift_for'
    m = re.match(r'Inspired by\s*(.*?)\s*by\s*(.*?)\.\s*', association_elem.text.strip())
    if m is not None:
        return 'inspired_by'

    raise NotImplementedError(f"no association implemented for {association_elem.text}")


# should process the jump regarding its type
def process_jump(jump_elem):
    return jump_elem.html


def process_work_notes(notes_wrapper):
    mod_type, heading, content = get_module_attrs(notes_wrapper)
    assert mod_type in ['notes', 'end_notes']
    associations_wrapper = notes_wrapper.query_selector("ul.associations")
    associations_html = associations_wrapper.html if associations_wrapper is not None else None

    jumps = [process_jump(j) for j in notes_wrapper.query_selector_all("p.jump")]

    return Ao3Notes(mod_type, heading, content, associations_html, jumps)


# I haven't seen associations in chapters yet
def process_chapter_notes(notes_wrapper):
    mod_type, heading, content = get_module_attrs(notes_wrapper)
    assert mod_type in ['notes', 'end_notes']
    jump_elems = [c for c in notes_wrapper.child_nodes if c.tag == 'p']
    jumps = [process_jump(j) for j in jump_elems] if len(jump_elems) > 0 else []

    return Ao3Notes(mod_type, heading, content, None, jumps)
