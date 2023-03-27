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


# NOTE: Manually enable this briefly if needed, but otherwise keep it skipped.
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

    _LOG_MESSAGE_PATTERN = re.compile(
        r'Backing off _post_request\(\.\.\.\) for [0-9.]+s '
        r'\(embed\._RateLimitError\)',
    )

    def setUp(self):
        """Help us avoid running the test on CI, and decrease stack size."""
        if os.getenv('CI') is not None:
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
        def run(thread_index):
            for loop_index in range(7):
                # Note: We support Python 3.7, so can't write {thread_index=}.
                embed.embed_one_req(
                    'Testing rate limiting. '
                    f'thread_index={thread_index} loop_index={loop_index}',
                )

        threads = [
            threading.Thread(target=run, args=(thread_index,))
            for thread_index in range(600)
        ]

        with self.assertLogs() as log_context:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        log_has_backoff_message = any(
            self._LOG_MESSAGE_PATTERN.fullmatch(record.message)
            for record in log_context.records
        )
        self.assertTrue(log_has_backoff_message)


if __name__ == '__main__':
    unittest.main()
