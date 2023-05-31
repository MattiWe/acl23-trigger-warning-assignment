from dataclasses import dataclass


@dataclass
class Ao3Tag:

    name: str
    path: str


def process_tag_wrapper(tag_wrapper):
    tags = tag_wrapper.query_selector_all("ul.commas > li > a.tag")
    return [
        Ao3Tag(name=t.text, path=t['href'])
        for t in tags
    ]
