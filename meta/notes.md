# Notes

## `TestDiskCachedEmbedOne` vs. `TestDiskCachedEmbedMany`

Those two test classes differ in four ways:

1. Which embedding functions the class is parameterized by.

2. What input (one text vs. a sequence of texts) is given to the functions.

3. What filename the embeddings should save to and load from.

4. When we test that a saved embedding can be loaded by "all three"
   implementations, *which* three implementations.

## Possible ways to reorganize the whole test suite

Reorganization may include how the suite is broken up into modules, but also
whether inheritance would better express the code reuse and fixture logic
currently done with `@parameterized_class` and helpers. In particular:

1. Overriding an abstract func property would automatically play well with
   monkey-patching, because expressions in property getters are evaluated when
   properties are accessed: during the tests, while any patching is active.
   This would eliminate the need for `_helpers.Caller`.

2. Inheritance could reuse test logic from classes in `test_embed` for the
   disk-caching versions in this module, to test behaviors expected both of
   them and the non-caching versions. Both hit and miss scenarios (which are
   alike in all the guaranteed behaviors they also share with non-caching
   versions) could be covered, by using different fixures. We could instead do
   this by expanding the existing `@parameterized_class` decoration. But the
   resulting test classes would be in `test_embed`, and the supporting fixture
   logic might be tricky to read and understand.

3. One or more base classes could handle all the fixture logic currently
   expressed with helper functions including our custom decorators. Most
   significantly, this could handle conditionally patching `embed.embed_*` to
   do in-memory caching, replacing `@maybe_cache_embeddings_in_memory`.
