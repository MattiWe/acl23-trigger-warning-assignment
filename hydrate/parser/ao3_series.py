from dataclasses import dataclass
import re
import logging


@dataclass
class Ao3Series:

    series_id: int
    name: str
    path: str


class Ao3ReferenceError(Exception):

    def __init__(self, reference):
        super().__init__(f"{reference} doesn't match the reference format")
        self.reference = reference


def process_series(series_span):
    info_span = series_span.query_selector("span.position")
    part = re.match(r'Part (\d+) of .*?', info_span.text.strip()).group(1)
    path = info_span.query_selector("a")['href']
    series_id_match = re.match(r'(?:https://archiveofourown\.org){0,1}/series/(\d+)', path)
    if series_id_match is None:
        raise Ao3ReferenceError(path)
    series_id = int(series_id_match.group(1))
    name = info_span.query_selector("a").text

    return {'series': Ao3Series(series_id, name, path), 'part': part}


# wrapper from endnotes
def process_series_module(series_wrapper):
    series_list = []
    for s in series_wrapper.query_selector_all("ul > li > span.series"):
        series = process_series(s)
        try:
            series_list.append(series)
        except Ao3ReferenceError as ao3_ref_err:
            logging.warning(f"{ao3_ref_err.reference} doesn't match the series reference format!")

    return series_list
