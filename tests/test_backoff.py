#!/usr/bin/env python

"""Specialized backoff testing."""

# pylint: disable=missing-function-docstring

import os
import re
import threading
import unittest

import embed


# NOTE: Manually enable this briefly if needed, but otherwise keep it skipped.
@unittest.skip("No need to regularly slam OpenAI's servers. Also: very slow.")
class TestBackoff(unittest.TestCase):
    """
    Tests that backoff works in the ``requests`` version.

    This is hard to check for, if one's OpenAI account is not subject to
    reduced rate limits. (Reduced rate limits are only in the trial period and
    shortly thereafter.) But occasionally it may be valuable to test rate
    limiting explicitly. So this sends a lot of requests to the OpenAI
    embeddings endpoint in a short time.
    """

    _LOG_MESSAGE_PATTERN = re.compile(
        r'INFO:backoff:Backing off _post_request(\.\.\.) for [0-9.]+s '
        r'\(embed\._RateLimitError\)',
    )

    def setUp(self):
        """Reduce the risk of accidentally running this on CI."""
        if os.getenv('CI') is not None:
            # pylint: disable=broad-exception-raised
            #
            # To signal a failure keeping the test from running at all, we
            # raise a direct Exception instance, which code under test should
            # never raise. (A more specific type would risk being misunderstood
            # as a specific error related to the code under test.)
            raise Exception(
                "These tests shouldn't run via continuous integration.")

    def test_embed_one_req_backs_off(self):
        def run(thread_index):
            for loop_index in range(100):
                # Note: We support Python 3.7, so can't write {thread_index=}.
                embed.embed_one_req(
                    'Testing rate limiting. '
                    f'thread_index={thread_index} loop_index={loop_index}',
                )

        threads = [
            threading.Thread(target=run, args=(thread_index,))
            for thread_index in range(100)
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
