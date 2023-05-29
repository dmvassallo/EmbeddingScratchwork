# How `tce-miss.log` and `tce-hit.log` are generated

## `tce-miss.log`

<!-- markdownlint-capture -->
<!-- markdownlint-disable line-length -->
```sh
TESTS_LOGGING_LEVEL=INFO python -m unittest -v tests/test_cached_embeddings.py -k TestDiskCacheMiss &>tce-miss.log
```
<!-- markdownlint-restore -->

## `tce-hit.log`

<!-- markdownlint-capture -->
<!-- markdownlint-disable line-length -->
```sh
TESTS_LOGGING_LEVEL=INFO python -m unittest -v tests/test_cached_embeddings.py -k TestDiskCacheHit &>tce-hit.log
```
<!-- markdownlint-restore -->
