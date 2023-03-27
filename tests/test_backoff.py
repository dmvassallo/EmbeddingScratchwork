#!/usr/bin/env python

"""
Specialized backoff testing.

This does end-to-end testing of rate limiting to check that backoff appears to
work as intended. The traffic generated is much greater than from the other
tests, so these tests are skipped by default and should not be run on CI.

To keep the traffic from being six times greater, only ``embed_one_req`` is
tested. (This differs from the tests in ``test_embed``, which test all six
functions, as should usually be done.) It's a reasonable tradeoff because:

1. Only ``embed_one_req`` and ``embed_many_req`` do backoff at all differently
   from the ways shown in https://platform.openai.com/docs/guides/rate-limits.

2. The way they do it may change, and tests are helpful for that. The way the
   other ``embed_`` functions in this project do backoff is unlikely to change.

3. They share their backoff logic. So it may be enough to test just one.
"""

import os
import re
import threading
import unittest

import embed

_THREAD_COUNT = 600
"""Number of concurrent threads making requests in the backoff test."""

_ITERATION_COUNT = 7
"""Number of requests each thread makes sequentially in the backoff test."""

_states_backoff = re.compile(
    r'Backing off _post_request\(\.\.\.\) for [0-9.]+s '
    r'\(embed\._RateLimitError\)',
).fullmatch
"""Check if the string is a log message about backoff with expected details."""


class _RequestThread(threading.Thread):
    """Thread for testing concurrent requests."""

    @classmethod
    def create_all(cls):
        """Create ``_THREAD_COUNT`` threads with distinct thread indices."""
        return [cls(thread_index) for thread_index in range(_THREAD_COUNT)]

    def __init__(self, thread_index):
        """Create a thread for testing backoff, with the given thread index."""
        super().__init__(name=f'Thread with index {thread_index}')
        self._thread_index = thread_index

    def run(self):
        """Call ``embed_one_req``, ``_ITERATION_COUNT`` times."""
        for loop_index in range(_ITERATION_COUNT):
            # Note: We support Python 3.7, so we can't write {loop_index=}.
            embed.embed_one_req(
                'Testing rate limiting. '
                f'thread_index={self._thread_index} loop_index={loop_index}',
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
                "These tests shouldn't run via continuous integration.")

        self._old_stack_size = threading.stack_size(32_768)

    def tearDown(self):
        """Restore the stack size."""
        threading.stack_size(self._old_stack_size)

    def test_embed_one_req_backs_off(self):
        """``embed_one_req`` backs off under high load and logs that it did."""
        threads = _RequestThread.create_all()

        with self.assertLogs() as log_context:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertTrue(
            any(_states_backoff(message) for message in log_context.output),
        )


if __name__ == '__main__':
    unittest.main()
