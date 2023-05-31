import re
from typing import Any
from dataclasses import dataclass
from datetime import datetime, date

from resiliparse.parse.html import HTMLTree
import logging

from .ao3_chapter import Ao3Chapter, process_chapter
from .ao3_page import check_for_errors
from .ao3_author import Ao3Author, process_byline
from .ao3_tag import Ao3Tag, process_tag_wrapper
from .ao3_comment import Ao3Comment, process_comments
from .ao3_module import get_module_attrs
from .ao3_notes import Ao3Notes, process_work_notes
from .ao3_summary import Ao3Summary, process_summary
from .ao3_series import Ao3Series, process_series_module, process_series, Ao3ReferenceError
from .ao3_collection import Ao3Collection, process_collections


class Ao3UnrevealedError(Exception):

    def __init__(self, work_id):
        super().__init__(f"The work {work_id} hasn't been revealed when it was crawled")
        self.work_id = work_id


def check_work_unrevealed(work_parser):
    notices = work_parser.query_selector_all("p.notice")
    for notice in notices:
        m = re.match(
            r'This work is part of an ongoing challenge and will be revealed soon! You can find details here:.*',
            notice.text.strip())
        if m is not None:
            return True
    return False


@dataclass
class Ao3Work:
    work_id: int
    tags: list[Ao3Tag]
    language: str
    collections: list[Ao3Collection]
    series: Any
    number_of_chapters: int
    number_of_chapters_planned: int
    number_of_words: int
    number_of_kudos: int
    number_of_comments: int
    number_of_bookmarks: int
    number_of_hits: int
    published_at: date
    updated_at: date
    title: str
    authors: list[Ao3Author]
    summary: Ao3Summary
    notes: Ao3Notes
    chapters: list[Ao3Chapter]
    endnotes: Ao3Notes
    children: Any
    kudos_from: Any
    comments: list[Ao3Comment]


def build_work(url, html):
    parser = HTMLTree.parse(html).body
    check_for_errors(parser)
    if check_work_unrevealed(parser):
        raise Ao3UnrevealedError(work_id=process_work_url(url))
    meta_wrapper = parser.query_selector("div#inner > div#main > div.wrapper")
    feedback_wrapper = parser.query_selector("div#inner > div#main div#feedback.feedback")

    tags, language, collections, series_from_meta, stats = process_work_meta(meta_wrapper)

    n_chapters, n_chapters_planned = stats.pop('chapters', (None, None))
    n_words = stats.pop('words', "0").replace(',', '')
    n_kudos = stats.pop('kudos', "0").replace(',', '')
    n_comments = stats.pop('comments', "0").replace(',', '')
    n_bookmarks = stats.pop('bookmarks', "0").replace(',', '')
    n_hits = stats.pop('hits', "0").replace(',', '')
    published = stats.pop('published', None)
    published = datetime.strptime(published.strip(), '%Y-%m-%d') if published is not None else None
    updated = stats.pop('status',
                        None)  # updated date has key 'status', but can also be completed under the same key -> flag?
    updated = datetime.strptime(updated.strip(), '%Y-%m-%d') if updated is not None else None
    n_words = int(n_words) if not (n_words is None or n_words == '') else None
    n_kudos = int(n_kudos) if n_kudos is not None else n_kudos
    n_comments = int(n_comments) if n_comments is not None else n_comments
    n_bookmarks = int(n_bookmarks) if n_bookmarks is not None else n_bookmarks
    n_hits = int(n_hits) if n_hits is not None else n_hits
    assert stats == {}, f"stats dictionary still has the following keys: {list(stats.keys())}"

    preface_wrapper = parser.query_selector("div#inner > div#main > div#workskin > div.preface.group:not(.afterword)")
    title, authors, summary, notes = process_work_content_preface(preface_wrapper)
    chapters_wrapper = parser.query_selector("div#inner > div#main > div#workskin > div#chapters")
    chapters = process_work_content_chapters(chapters_wrapper)
    endnotes_module = parser.query_selector("div#inner > div#main > div#workskin div#work_endnotes.end.notes.module")
    endnotes = process_work_notes(endnotes_module) if endnotes_module is not None else None
    series_module = parser.query_selector("div#inner > div#main > div#workskin div#series.series.module")
    series_from_endnotes = process_series_module(series_module) if series_module is not None else []
    children_module = parser.query_selector("div#inner > div#main > div#workskin div#children.children.module")
    children = process_children_module(children_module) if children_module is not None else []

    kudos_from, comments, cmts_pages = process_work_feedback(feedback_wrapper)

    return Ao3Work(
        work_id=int(process_work_url(url)),
        tags=tags,
        language=language,
        collections=collections,
        series=series_from_endnotes,
        number_of_chapters=n_chapters,
        number_of_chapters_planned=n_chapters_planned,
        number_of_words=n_words,
        number_of_kudos=n_kudos,
        number_of_comments=n_comments,
        number_of_bookmarks=n_bookmarks,
        number_of_hits=n_hits,
        published_at=published,
        updated_at=updated,
        title=title,
        authors=authors,
        summary=summary,
        notes=notes,
        chapters=chapters,
        endnotes=endnotes,
        children=children,
        kudos_from=kudos_from,
        comments=comments,
    )


def process_work_url(work_url):
    m = re.match(r'https://archiveofourown.org/works/(\d+)', work_url)
    if m is None:
        return None
    work_id = m.group(1)
    return work_id


def process_stats_wrapper(stats_wrapper):
    stats = {}
    for stat in stats_wrapper.query_selector_all("dl.stats > dd"):
        stat_type = stat.class_list[0]
        stats[stat_type] = stat.text

    present, planned = stats['chapters'].split('/')
    present = int(present)
    planned = int(planned) if not planned == '?' else None
    stats['chapters'] = (present, planned)

    return stats


def process_work_meta(meta):
    inner_meta = meta.query_selector("dl.work.meta.group")
    tags = {
        'rating': [],
        'warning': [],
        'category': [],
        'fandom': [],
        'relationship': [],
        'character': [],
        'freeform': [],
    }
    for t in inner_meta.query_selector_all("dd.tags"):
        tag_type = t.class_list[0]
        tags[tag_type] += process_tag_wrapper(t)
    tags['additional'] = tags.pop('freeform')
    languages = [l.text.strip() for l in inner_meta.query_selector_all("dd.language")]
    collections_wrapper = inner_meta.query_selector("dd.collections")
    collections = process_collections(collections_wrapper) if collections_wrapper else []

    series = []
    for s in inner_meta.query_selector_all("dd.series > span.series"):
        try:
            series.append(process_series(s))
        except Ao3ReferenceError as ao3_ref_err:
            logging.warning(f"{ao3_ref_err.reference} doesn't match the series reference format!")

    stats = process_stats_wrapper(inner_meta.query_selector("dd.stats"))

    assert len(languages) <= 1, "found more than one language"
    language = languages[0] if len(languages) > 0 else None
    return tags, language, collections, series, stats


def process_work_content_preface(preface_content):
    title = preface_content.query_selector("h2.title.heading").text.strip()
    authors_byline = preface_content.query_selector("h3.byline.heading")
    authors = process_byline(authors_byline)
    summary = None
    notes = None
    for mod in (c for c in preface_content.child_nodes if c.tag == "div" and "module" in c.class_list):
        mod_type, _, _ = get_module_attrs(mod)
        if mod_type == 'summary':
            summary = process_summary(mod)
        elif mod_type == 'notes':
            notes = process_work_notes(mod)
        else:
            raise NotImplementedError(f"unknown module ({mod_type})!")

    return title, authors, summary, notes


def process_work_content_chapters(chapters_content):
    chapters = []
    # case work has multiple chapters
    for i, chapter_wrap in enumerate(chapters_content.query_selector_all("div.chapter[id^=chapter-]")):
        chapter_num = i + 1
        chapters.append(process_chapter(chapter_wrap, chapter_num))
    # case work has only 1 chapter
    if len(chapters) < 1:
        chapter_content = chapters_content.query_selector("div.userstuff")
        chapter_content_data = chapter_content.html if chapter_content is not None else None
        chapters.append(
            Ao3Chapter(title=None, chapter_index=None, path=None, summary=None, notes=None,
                       content=chapter_content_data, endnotes=None))
    return chapters


# ignores restricted works (see https://archiveofourown.org/works/39?view_adult=true&view_full_work=true&show_comments=true)
def process_children_module(children_wrapper):
    children = []
    for c in children_wrapper.query_selector_all("ul > li"):
        refs = c.query_selector_all("a")
        if not refs:
            continue
        work_ref = refs[0]['href']
        m = re.match(r'/works/(\d+)', work_ref)
        if m is not None:
            children.append(m.group(1))
    return children


def process_work_content_afterword(afterword_wrapper):
    endnotes = None
    series = []
    children = []
    for mod in (c for c in afterword_wrapper.child_nodes if c.tag == "div" and "module" in c.class_list):
        mod_type, _, _ = get_module_attrs(mod)
        if mod_type == 'end_notes':
            if endnotes is None:
                endnotes = process_work_notes(mod)
            else:
                raise NotImplementedError("more than one endnotes module!")
        elif mod_type == 'series':
            series += process_series_module(mod)
        elif mod_type == 'children':
            children += process_children_module(mod)
        else:
            raise NotImplementedError(f"unknown module ({mod_type})!")

    return endnotes, series, children


def process_kudos(kudos_wrapper):
    kudos_p = kudos_wrapper.query_selector("p.kudos")
    if kudos_p is None:
        return None, None, None
    kudos_from = []
    n_kudos_from_other_users = 0
    n_kudos_from_guests = 0
    for u in kudos_p.query_selector_all("a"):
        name = u.text
        ref = u['href']
        m = re.match(r'/users/(\w+)', ref)
        if m is not None:
            kudos_from.append({'name': name, 'ref': ref})
            continue
        m = re.match(r'(\d+) more users', name)
        if m is not None:
            n_kudos_from_other_users = int(m.group(1))
    m = re.search(r'(\d+) guests\s+left\s+kudos\s+on\s+this\s+work!', kudos_p.text.strip())
    if not m is None:
        n_kudos_from_guests = int(m.group(1))

    return kudos_from, n_kudos_from_other_users, n_kudos_from_guests


def process_work_feedback(feedback):
    kudos_wrapper = feedback.query_selector("div#kudos")
    kudos_from, from_other_users, from_guests = process_kudos(kudos_wrapper) if kudos_wrapper is not None else (
    None, None, None)
    comments_wrapper = feedback.query_selector("div#comments_placeholder")
    cmts_pages, comments = process_comments(comments_wrapper) if comments_wrapper is not None else (None, None)
    return kudos_from, comments, cmts_pages
