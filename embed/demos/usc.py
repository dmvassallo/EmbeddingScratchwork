"""Semantic search in the U.S. Code."""

__all__ = [
    'USC_STEM',
    'extract_usc',
    'drop_attributes',
    'serialize_xml_clean',
    'count_tokens_xml',
    'count_tokens_xml_clean',
    'tabulate_token_counts',
    'with_totals',
    'with_cost_columns',
    'full_tabulate_token_counts',
    'show_tails',
    'show_wrapped',
    'get_embeddable_elements',
    'is_repealed',
]

import copy
import logging
import os
from pathlib import Path
import shutil
import textwrap
from zipfile import ZipFile

from lxml import etree as ET
import polars as pl

import embed

USC_STEM = 'xml_uscAll@118-3not328'
"""Directory name and XML file basename used for U.S. Code files."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule (``embed.demos.usc``)."""


# FIXME: Check if anything like CVE-2007-4559 applies to zip files.
def extract_usc(data_dir):
    """Extract the U.S. Code. If the directory already exists, do nothing."""
    # Do nothing if a directory with the expected name already exists.
    subdir = Path(data_dir, USC_STEM)
    if subdir.is_dir():
        return

    # Stop and report failure if the archive does not exist.
    archive_path = Path(data_dir, f'{USC_STEM}.zip')
    if not archive_path.is_file():
        raise FileNotFoundError(f'no archive: {archive_path}')

    # Create the target directory and extract the archive into it.
    os.mkdir(subdir)
    with ZipFile(archive_path, mode='r') as archive:
        archive.extractall(path=subdir)

    # Eliminate extra nesting, if the archive has its own same-named directory.
    subsubdir = subdir / USC_STEM
    if subsubdir.is_dir():
        for entry in subsubdir.iterdir():
            shutil.move(entry, subdir)
        os.rmdir(subsubdir)


def drop_attributes(element_text):
    """Drop attributes from all XML tags in a string."""
    tree = ET.fromstring(element_text.encode('utf-8'))
    for element in tree.iter():
        element.attrib.clear()
    return ET.tostring(tree, encoding='unicode')


def serialize_xml_clean(element):
    """Copy an XML subtree, omitting attributes from all tags."""
    dup = copy.deepcopy(element)
    for subelement in dup.iter():
        subelement.attrib.clear()
    return ET.tostring(dup, encoding='unicode')


def count_tokens_xml(element):
    """Count the tokens in an XML element (node or whole tree)."""
    text = ET.tostring(element, encoding='unicode')
    return embed.count_tokens(text)


def count_tokens_xml_clean(element):
    """Count the tokens in an XML element (node or whole tree)."""
    cleaned = serialize_xml_clean(element)
    return embed.count_tokens(cleaned)


def _build_row(path):
    """Build a table row with just Title, Tokens, and Clean Tokens columns."""
    full_text = path.read_text(encoding='utf-8')

    return {
        'Title': path.stem,
        'Tokens': embed.count_tokens(full_text),
        'Clean Tokens': embed.count_tokens(drop_attributes(full_text)),
    }


def tabulate_token_counts(data_dir, *, filename='usc_token_counts.csv'):
    """
    Get a table of tokens counts in each USC title, as a Polars data frame.

    If a saved table is found, it is assumed current and returned. Otherwise,
    the XML files' tokens are counted and the table is generated. This may use
    a significant amount of memory, because tokens are counted by tokenizing
    entire titles of the U.S. Code with ``tiktoken`` and counting the tokens,
    and some titles are very long.
    """
    token_table_path = Path(data_dir, filename)
    try:
        # Try to load a table of token counts that has already been saved.
        df_counts = pl.read_csv(token_table_path)
    except OSError:
        # Extract the files if they do not appear to have been extracted.
        extract_usc(data_dir)

        # Find paths to relevant XML files, failing clearly if none were found.
        paths = sorted((data_dir / USC_STEM).glob('*.xml'))
        assert paths, 'No XML files found.'

        # Read the files and count tokens with and without tag attributes.
        df_counts = pl.DataFrame(_build_row(path) for path in paths)

        # Save the results and check that they were saved correctly.
        df_counts.write_csv(token_table_path)
        assert pl.read_csv(token_table_path).frame_equal(df_counts)

    return df_counts


def with_totals(df_without_totals):
    """Copy a token count table and add a row of totals at the bottom."""
    totals = (
        df_without_totals
        .sum()
        .with_columns(pl.Series("Title", ["TOTALS"]))
    )
    return pl.concat([df_without_totals, totals])


def with_cost_columns(df_without_costs, token_cost):
    """
    Copy a token count table and add Cost and Clean Cost columns to it.

    Since ``token_cost`` is a monetary value, the ``decimal.Decimal`` type is
    recommended (but not required).
    """
    return df_without_costs.with_columns(
        pl.col('Tokens')
          .apply(token_cost.__mul__)
          .alias('Cost ($)'),
        pl.col('Clean Tokens')
          .apply(token_cost.__mul__)
          .alias('Clean Cost ($)'),
    ).select('Title', 'Tokens', 'Cost ($)', 'Clean Tokens', 'Clean Cost ($)')


def full_tabulate_token_counts(data_dir, token_cost):
    """
    Build a default augmented token count table with costs and totals.

    See ``tabulate_token_counts``, which provides most of the underlying
    functionality. For more control, call that and augment (see ``with_totals``
    and ``with_cost_columns``) and/or filter the result as desired.
    """
    df_counts = tabulate_token_counts(data_dir)
    df_with_totals = with_totals(df_counts)
    return with_cost_columns(df_with_totals, token_cost)


def show_tails(root):
    """Print a report of all elements' tail text in an XML tree."""
    for _, element in ET.iterwalk(root, events=('start',)):
        if element.tail and element.tail.strip():
            print(f'{element}: {element.tail!r}')


def show_wrapped(element, *, width=140, limit=None):
    """Print the content of an element, hard-wrapped for readability."""
    element_text = ET.tostring(element, encoding='unicode')
    display_text = element_text if limit is None else element_text[:limit]
    print('\n'.join(textwrap.wrap(display_text, width=width)))


# FIXME: (1) Don't lose text appearing directly in elements whose subelements
#            we traverse to!
#
#        (2) Avoid traversing into elements like <em> that are not conceptually
#            logical portions of the U.S. Code.
#
def get_embeddable_elements(section):
    """Break up an XML tree into elements that are small enough to embed."""
    selection = []
    iterator = ET.iterwalk(section, events=('start',))

    for _, element in iterator:
        token_count = count_tokens_xml_clean(element)

        if token_count <= embed.CONTEXT_LENGTH:  # We can embed this subtree.
            iterator.skip_subtree()
            selection.append(element)
        elif len(element) == 0:  # We're at the bottom and it's still too big.
            _logger.error('Too-big leaf %r (%d tokens).', element, token_count)
        else:
            pass  # FIXME: Check for text this element directly contains.

    return selection


def is_repealed(element):
    """Check if a section or other element of the USC is marked repealed."""
    return element.get('status') == 'repealed'
