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
import pooch

import embed

USC_STEM = 'xml_uscAll@118-6'
"""Directory name and XML file basename used for U.S. Code files."""

# FIXME: Change this to a fast GitHub mirror with a better compressed archive.
_URL_PREFIX = 'https://uscode.house.gov/download/releasepoints/us/pl/118/6/'
"""URL to a remote directory where the U.S. Code can be downloaded."""

_ARCHIVE_HASH = (
    'sha256:e9c6e8063a4151ce6dc68ee0fcc3adc6e1ec3e5a24f431dab2d6b40ab8f4370d'
)
"""SHA256 hash of the U.S. Code archive available in ``_URL_PREFIX``."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this submodule (``embed.demos.usc``)."""


def _download_usc(data_dir, archive_filename):
    """Download the U.S. Code."""
    pooch.retrieve(
        url=(_URL_PREFIX + archive_filename),
        known_hash=_ARCHIVE_HASH,
        fname=archive_filename,
        path=data_dir,  # Path to the directory to put it in.
        progressbar=True,
    )


def _require_usc_archive(data_dir, download):
    """
    If the archive is absent, either fail or download it, as appropriate.

    Raises various exceptions on failure. Returns the archive path on success.
    """
    archive_filename = f'{USC_STEM}.zip'
    archive_path = Path(data_dir, archive_filename)

    if not archive_path.is_file():
        if not download:
            raise FileNotFoundError(f'no archive: {archive_path}')
        _download_usc(data_dir, archive_filename)

    return archive_path


def _do_extract_usc(subdir, archive_path):
    """
    Create a target directory, extract the USC, and eliminate extra nesting.

    This does the actual extraction for ``extract_usc``, which calls it after
    performing some preparatory steps.
    """
    # Create the target directory.
    os.mkdir(subdir)

    # Extract the archive to the target directory.
    with ZipFile(archive_path, mode='r') as archive:
        archive.extractall(path=subdir)

    # Eliminate extra nesting, if the archive has its own same-named directory.
    subsubdir = subdir / USC_STEM
    if subsubdir.is_dir():
        for entry in subsubdir.iterdir():
            shutil.move(entry, subdir)
        os.rmdir(subsubdir)


# FIXME: Check if anything like CVE-2007-4559 applies to zip files.
def extract_usc(data_dir, *, download=False):
    """
    Extract the U.S. Code. If the directory already exists, do nothing.

    If ``download=True`` and the archive file does not exist (and neither does
    the directory), then a U.S. Code archive will be downloaded and extracted.
    """
    # Do nothing if a directory with the expected name already exists.
    subdir = Path(data_dir, USC_STEM)
    if subdir.is_dir():
        return

    # Ensure the archive exists, or raise an exception.
    archive_path = _require_usc_archive(data_dir, download)

    # Actually extract the archive.
    _do_extract_usc(subdir, archive_path)


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


# FIXME: Avoid breaking up elements like <em> that are not, in a conceptual
#        sense, specific logical portions of the U.S. Code.
def get_embeddable_elements(section, *, strict=True):
    """Break up an XML tree into elements that are small enough to embed."""
    selection = []
    lost_leaves = lost_texts = lost_tails = 0
    iterator = ET.iterwalk(section, events=('start',))

    for _, element in iterator:
        token_count = count_tokens_xml_clean(element)

        if token_count <= embed.CONTEXT_LENGTH:
            # We can embed this subtree.
            iterator.skip_subtree()
            selection.append(element)
        elif len(element) == 0:
            # We're at the bottom and it's still too big.
            _logger.error('Too-big leaf %r (%d tokens).', element, token_count)
            lost_leaves += 1
        else:
            if element.text and element.text.strip():
                # We lose this element's "text" text by traversing further.
                _logger.error('%s: lost text: %r', element.tag, element.text)
                lost_texts += 1
            if element.tail and element.tail.strip():
                # We lose this element's "tail" text by traversing further.
                _logger.error('%s: lost tail: %r', element.tag, element.tail)
                lost_tails += 1

    if strict and (lost_leaves != 0 or lost_texts != 0 or lost_tails != 0):
        raise ValueError('selection loses content '
                         f'({lost_leaves=}, {lost_texts=}, {lost_tails=})')

    return selection


def is_repealed(element):
    """Check if a section or other element of the USC is marked repealed."""
    return element.get('status') == 'repealed'
