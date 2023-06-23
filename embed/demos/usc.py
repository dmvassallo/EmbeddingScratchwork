"""Semantic search in the U.S. Code."""

__all__ = [
    'USC_STEM',
    'extract_usc',
    'drop_attributes',
    'show_tails',
]

from pathlib import Path
from zipfile import ZipFile

from lxml import etree as ET

USC_STEM = 'xml_uscAll@118-3not328'
"""Directory name and XML file basename used for U.S. Code files."""


# FIXME: Check if anything like CVE-2007-4559 applies to zip files.
#
# FIXME: Handle how the official USC archive contains "loose" top-level files
#        rather than a top-level directory.
#
def extract_usc(data_dir):
    """Extract the U.S. Code. If the directory already exists, do nothing."""
    if not Path(data_dir, USC_STEM).is_dir():
        with ZipFile(data_dir / f'{USC_STEM}.zip', mode='r') as archive:
            archive.extractall(path=data_dir)


def drop_attributes(element_text):
    """Drop attributes from XML tags."""
    tree = ET.fromstring(element_text.encode('utf-8'))
    for element in tree.iter():
        element.attrib.clear()
    return ET.tostring(tree, encoding='unicode')


def show_tails(root):
    """Print a report of all elements' tail text in an XML tree."""
    for _, element in ET.iterwalk(root, events=('start',)):
        if element.tail and element.tail.strip():
            print(f'{element}: {element.tail!r}')
