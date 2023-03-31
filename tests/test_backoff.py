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

import embed

_STACK_SIZE = 32_768
"""Stack size in bytes for newly created worker threads. Do not change this."""

_BATCH_COUNT = 600
"""Maximum number of concurrent threads making requests in the backoff test."""

_BATCH_SIZE = 8
"""Number of requests each batch makes sequentially in the backoff test."""

_is_backoff_message = re.compile(
    r'INFO:backoff:Backing off _post_request\(\.\.\.\) for [0-9.]+s '
    r'\(<Response \[429\]>\)',
).fullmatch
"""Check if the string is a log message about backoff with expected details."""


def _run_batch(batch_index):
    """Run a batch of ``_BATCH_SIZE`` sequential jobs in the backoff test."""
    for job_index in range(_BATCH_SIZE):
        # Note: We support Python 3.7, so we can't write {batch_index=}.
        embed.embed_one_req(
            'Testing rate limiting. '
            f'batch_index={batch_index} job_index={job_index}',
        )


# NOTE: Manually enable this briefly if needed, but otherwise keep it skipped.
#
# TODO: After PR #56, run if the TESTS_RUN_BACKOFF_TEST_I_KNOW_WHAT_I_AM_DOING
#       environment variable is set to a truthy value (but still NEVER on CI).
#
@unittest.skip("No need to regularly slam OpenAI's servers. Also: very slow.")
class TestBackoff(unittest.TestCase):
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
        if 'CI' in os.environ:
            # pylint: disable=broad-exception-raised
            #
            # To signal a failure keeping the test from running at all, we
            # raise a direct Exception instance, which code under test should
            # never raise. (A more specific type would risk being misunderstood
            # as a specific error related to the code under test.)
            raise Exception(
                "This test shouldn't run via continuous integration.")

        self._old_stack_size = threading.stack_size(_STACK_SIZE)

        logging.warning(
            "Running full backoff test, which shouldn't usually be done.")

    def tearDown(self):
        """Restore the stack size."""
        threading.stack_size(self._old_stack_size)

    def test_embed_one_req_backs_off(self):
        """``embed_one_req`` backs off under high load and logs that it did."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=_BATCH_COUNT
                                                   ) as executor:
            with self.assertLogs() as log_context:
                # Make many requests. Raise any exceptions on the main thread.
                collections.deque(
                    executor.map(_run_batch, range(_BATCH_COUNT)),
                    maxlen=0,
                )

        got_backoff = any(
            _is_backoff_message(message)
            for message in log_context.output
        )
        self.assertTrue(got_backoff)


if __name__ == '__main__':
    unittest.main()
