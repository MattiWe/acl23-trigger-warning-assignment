import re
from typing import Any
from dataclasses import dataclass
from .ao3_author import Ao3Author
from datetime import datetime

from resiliparse.parse.html import NodeType


@dataclass
class Ao3Comment:

    comment_id: int
    author: Ao3Author
    refers_to_chapter: int | None
    content: str
    parent_comment: int
    parent_thread: int
    published_at: datetime
    timezone: Any


def process_work_comments_url(url):
    m = re.match(r'https://archiveofourown.org/works/(\d+)\?.*?page=(\d+).*', url)
    if m is None:
        return None
    work_id = m.group(1)
    comment_page = int(m.group(2))
    return work_id, comment_page
    

def process_comment(comment_wrapper, parent_id):
    author_wrap = comment_wrapper.query_selector("h4.heading.byline")
    if author_wrap is None: # case comment deleted
        return None
    if author_wrap.query_selector("a") is not None: # case ao3 user
        author_name = author_wrap.query_selector("a").text
        author_ref = author_wrap.query_selector("a")['href']
    else:  # case guest author
        author_name = ''.join([c.text for c in author_wrap.child_nodes if c.type == NodeType.TEXT]).strip()
        author_ref = None
    chapter_num = None
    if author_wrap.query_selector("span.parent") is not None: # case comment refers to chapter
        chapter_text = author_wrap.query_selector("span.parent").text.strip() 
        m = re.match(r'on Chapter (\d+)', chapter_text)
        if m is not None: # case chapter has a number
            chapter_num = int(m.group(1))
    author = Ao3Author(author_name, author_ref)
    content = comment_wrapper.query_selector("blockquote.userstuff").html
    comment_id = comment_wrapper.id.split('_')[-1]
    comment_date_wrap = author_wrap.query_selector("span.posted.datetime")
    c_day = int(comment_date_wrap.query_selector("span.date").text.strip())
    c_month = comment_date_wrap.query_selector("abbr.month").text.strip()
    c_year = int(comment_date_wrap.query_selector("span.year").text.strip())
    c_time = comment_date_wrap.query_selector("span.time").text.strip()
    c_timezone_title = comment_date_wrap.query_selector("abbr.timezone")['title']
    c_timezone_abbr = comment_date_wrap.query_selector("abbr.timezone").text.strip()
    c_dt_str = f"{c_day} {c_month} {c_year} {c_time}"
    c_date = datetime.strptime(c_dt_str, '%d %b %Y %I:%M%p')
    root_thread_id = None
    actions_wrappers = comment_wrapper.query_selector_all("ul.actions > li > a")
    root_thread_wrapper_list = [w for w in actions_wrappers if w.text == 'Parent Thread']
    assert len(root_thread_wrapper_list) <= 1
    if len(root_thread_wrapper_list) > 0:
        root_thread_wrapper = root_thread_wrapper_list[0]
        root_thread_ref = root_thread_wrapper['href']
        root_thread_id = re.match(r'/comments/(\d+)', root_thread_ref).group(1)
    
    return Ao3Comment(
        comment_id=comment_id,
        author=author,
        refers_to_chapter=chapter_num,
        content=content,
        parent_comment=parent_id,
        parent_thread=root_thread_id,
        published_at=c_date,
        timezone={'title': c_timezone_title, 'abbreviation': c_timezone_abbr}
    )


def process_comments_thread(thread_wrapper, parent_comment_id):
    thread_comments = []
    last_comment_id = None
    for elem in thread_wrapper.child_nodes:

        if elem.tag == 'li' and len(elem.class_list) < 1:
            threads = [c for c in elem.child_nodes if c.tag == 'ol' and 'thread' in c.class_list]
            assert len(threads) == 1
            thread_comments += process_comments_thread(threads[0], last_comment_id)

        elif elem.tag == 'li' and 'group' in elem.class_list and 'comment' in elem.class_list:
            comment = process_comment(elem, parent_comment_id)
            if comment is None:
                last_comment_id = parent_comment_id
                continue
            thread_comments.append(comment)
            last_comment_id = comment.comment_id
    
    return thread_comments


def process_comments(comments_wrapper):
    comments_pages = 1
    pagination = comments_wrapper.query_selector("ol.pagination.actions")
    if pagination is not None:
        comments_pages = int(pagination.query_selector_all("li:not(.previous):not(.gap):not(.next) > a")[-1].text)
    root_threads = [c for c in comments_wrapper.child_nodes if c.tag == 'ol' and 'thread' in c.class_list]
    assert len(root_threads) <= 1
    comments = process_comments_thread(root_threads[0], None)
    
    return comments_pages, comments


if __name__=="__main__":
    pass