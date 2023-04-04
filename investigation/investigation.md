# Bug in cache redesign

## Producing the logs

To produce the logs (which I subsequently moved into this directory), I ran
this command with the previous commit checked out:

```bash
TESTS_LOGGING_LEVEL=DEBUG TESTS_CACHE_EMBEDDING_CALLS=yes tests/test_embed.py -v |& tee -a old.log
```

And this command with the current commit checked out:

```bash
TESTS_LOGGING_LEVEL=DEBUG TESTS_CACHE_EMBEDDING_CALLS_IN_MEMORY=yes tests/test_embed.py -v |& tee -a new.log
```

## Searching the log

To find the bug, I'm using this regular expression on the logs:

```text
test_\w+\b \(.+\) \.\.\.(?! DEBUG:root:In-memory cache)
```

And the modified regex without `?!`, to see where that *does* appear:

```text
test_\w+\b \(.+\) \.\.\.( DEBUG:root:In-memory cache)
```

(The `(` and `)` are retained there because it's easier to switch between them
in the editor without changing them.)

This *suggests*, somehow, that patching is happening on the `embed_one*`
functions but not the `embed_many*` functions.
