from dataclasses import dataclass


@dataclass
class Ao3Collection:

    name: str
    path: str


def process_collections(collection_wrapper):
    collections = []
    for c in [x for x in collection_wrapper.child_nodes if x.tag == 'a']:
        name = c.text.strip()
        path = c['href']
        collections.append(Ao3Collection(name, path))

    return collections
