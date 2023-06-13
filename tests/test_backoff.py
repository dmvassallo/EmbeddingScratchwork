#!/usr/bin/env python

"""
Specialized backoff testing.

This does end-to-end testing of rate limiting to check that backoff appears to
work as intended. The traffic generated is much greater than from the other
tests, so the test here is skipped by default and should not be run on CI.

To keep the traffic from being six times greater, only ``embed_one_req`` is
tested. (This differs from the tests in ``test_embed``, which test all six
functions, as should usually be done.) It's a reasonable tradeoff because:

1. Only ``embed_one_req`` and ``embed_many_req`` do backoff at all differently
   from the ways shown in https://platform.openai.com/docs/guides/rate-limits.

2. The way they do it may change, and tests are helpful for that. The way the
   other ``embed_`` functions in this project do backoff is unlikely to change.

3. They share their backoff logic. So it may be enough to test just one.
"""

import collections
import concurrent.futures
import logging
import os
import re
import threading
import unittest

import backoff

import embed
from tests import _bases, _helpers

_STACK_SIZE = 32_768
"""Stack size in bytes for newly created worker threads. Do not change this."""

_BATCH_COUNT = 600
"""Maximum number of concurrent threads making requests in the backoff test."""

_BATCH_SIZE = 8
"""Number of requests each batch makes sequentially in the backoff test."""

_BACKOFF_MESSAGE_REGEX = re.compile(
    r'INFO:backoff:Backing off _post_request\(\.\.\.\) for [0-9.]+s '
    r'\(<Response \[429\]>\)',
)
"""Regex to match a log message aobut backoff with expected details."""

_logger = logging.getLogger(__name__)
"""Logger for messages from this test module."""


def _run_batch(batch_index):
    """Run a batch of ``_BATCH_SIZE`` sequential jobs in the backoff test."""
    for job_index in range(_BATCH_SIZE):
        text = f'Testing rate limiting. {batch_index=} {job_index=}'
        embed.embed_one_req(text)


@unittest.skipUnless(
    _helpers.getenv_bool('TESTS_RUN_BACKOFF_TEST_I_KNOW_WHAT_I_AM_DOING'),
    "No need to regularly slam OpenAI's servers. Also: very slow.",
)
class TestBackoff(_bases.TestBase):
    """
    Test backoff in one of the functions using ``requests`` (``test_one_req``).

    This can be hard to check for, if one's OpenAI account is not subject to
    reduced rate limits. (Rate limits for access to language models are only
    reduced during the trial period and shortly thereafter.) But occasionally
    it may be valuable to test rate limiting explicitly. So this sends a lot of
    requests to the OpenAI embeddings endpoint in a short time. Use sparingly.
    """

    def setUp(self):
        """Help us avoid running the test on CI, and decrease stack size."""
        super().setUp()

        if 'CI' in os.environ:
            message = "This test shouldn't run via continuous integration."
            raise RuntimeError(message)

        self._old_stack_size = threading.stack_size(_STACK_SIZE)

        _logger.warning(
            "Running full backoff test, which shouldn't usually be done.")

    def tearDown(self):
        """Restore the stack size."""
        threading.stack_size(self._old_stack_size)
        super().tearDown()

    def test_embed_one_req_backs_off(self):
        """``embed_one_req`` backs off under high load and logs that it did."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=_BATCH_COUNT
                                                   ) as executor:
            with self.assertLogs(logger=backoff.__name__) as log_context:
                # Make many requests. Raise any exceptions on the main thread.
                collections.deque(
                    executor.map(_run_batch, range(_BATCH_COUNT)),
                    maxlen=0,
                )

        got_backoff = any(
            _BACKOFF_MESSAGE_REGEX.fullmatch(message)
            for message in log_context.output
        )
        self.assertTrue(got_backoff)


if __name__ == '__main__':
    unittest.main()
