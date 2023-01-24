import re
from dataclasses import dataclass

from .ao3_summary import Ao3Summary, process_summary
from .ao3_notes import Ao3Notes, process_chapter_notes


@dataclass
class Ao3Chapter:

    title: str
    chapter_index: int
    path: str
    summary: Ao3Summary
    notes: Ao3Notes
    content: str
    endnotes: Ao3Notes

    def get_chapter_id(self):
        return re.match(r'/works/\d+?/chapters/(\d+)', self.path).group(1)


# die chapter prefaces sollten direkte Kindknoten sein
# Beispiel: https://archiveofourown.org/works/11080797?view_adult=true&view_full_work=true&show_comments=true
# dort ist eine work in dem userstuff (keine Ahnung wie das sein kann)
# ich gehe jetzt über ids von summary, notes und endnotes statt über prefaces
def process_chapter(chapter_wrapper, chapter_num):
    
    title_wrapper = chapter_wrapper.query_selector("div.chapter.preface.group h3.title")
    title = title_wrapper.text.strip()
    ref = title_wrapper.query_selector("a")['href']
    chapter_notes_wrapper = chapter_wrapper.query_selector("div#notes.notes.module")
    notes = process_chapter_notes(chapter_notes_wrapper) if chapter_notes_wrapper is not None else None
    chapter_summary_wrapper = chapter_wrapper.query_selector("div#summary.summary.module")
    summary = process_summary(chapter_summary_wrapper) if chapter_summary_wrapper is not None else None
    chapter_endnotes_wrapper = chapter_wrapper.query_selector("div.end.notes.module[id^=chapter][id$=endnotes]")
    endnotes = process_chapter_notes(chapter_endnotes_wrapper) if chapter_endnotes_wrapper is not None else None
    
    chapter_content_wrapper = chapter_wrapper.query_selector("div.userstuff.module")
    chapter_content_data = chapter_content_wrapper.html if chapter_content_wrapper is not None else None

    # besser die Kapitelnummer über die id holen
    # Sonst gibt es Dopplungen z.B. in Kapitel 44 bei https://archiveofourown.org/works/11094339?view_adult=true&view_full_work=true&show_comments=true
    # chapter_num = re.match(r'Chapter (\d+)', title).group(1)
    # es gibt aber auch so Dopplungen (siehe https://archiveofourown.org/works/11660130?view_adult=true&view_full_work=true&show_comments=true)
    html_chapter_num_str = chapter_wrapper.id.split('-')[-1]
    html_chapter_num = int(html_chapter_num_str) if html_chapter_num_str else None 
    # also muss die Kapitelnummer von außen übergeben werden

    return Ao3Chapter(
        title=title,
        chapter_index=html_chapter_num,
        path=ref,
        summary=summary,
        notes=notes,
        content=chapter_content_data,
        endnotes=endnotes
    )


if __name__ == "__main__":
    pass