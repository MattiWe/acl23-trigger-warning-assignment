import re
from dataclasses import dataclass
import urllib.parse


@dataclass
class Ao3Author:
    name: str
    path: str


# set to prevent adding the same author multiple times
# e.g. https://archiveofourown.org/works/352322?view_adult=true&view_full_work=true&show_comments=true
def process_byline(byline_wrapper):
    author_tuples = {
        (a.text, a['href'])
        for a in byline_wrapper.query_selector_all("a")
    }
    
    return [Ao3Author(name, path) for name, path in author_tuples]
