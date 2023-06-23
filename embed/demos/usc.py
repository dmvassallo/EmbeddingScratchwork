"""Semantic search in the U.S. Code."""

__all__ = [
    'USC_STEM',
    'extract_usc',
    'drop_attributes',
    'show_tails',
    'tabulate_token_counts',
    'with_totals',
    'with_cost_columns',
    'full_tabulate_token_counts',
]

from pathlib import Path
from zipfile import ZipFile

from lxml import etree as ET
import polars as pl

from embed import count_tokens

USC_STEM = 'xml_uscAll@118-3not328'
"""Directory name and XML file basename used for U.S. Code files."""


# FIXME: Check if anything like CVE-2007-4559 applies to zip files.
#
# FIXME: Handle how the official USC archive contains "loose" top-level files
#        rather than a top-level directory.
#
def extract_usc(data_dir):
    """Extract the U.S. Code. If the directory already exists, do nothing."""
    if Path(data_dir, USC_STEM).is_dir():
        return
    archive_path = Path(data_dir, f'{USC_STEM}.zip')
    with ZipFile(archive_path, mode='r') as archive:
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


def _build_row(path):
    """Build a table row with just Title, Tokens, and Clean Tokens columns."""
    full_text = path.read_text(encoding='utf-8')

    return {
        'Title': path.stem,
        'Tokens': count_tokens(full_text),
        'Clean Tokens': count_tokens(drop_attributes(full_text)),
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
