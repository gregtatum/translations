# Segmentation in JS and general segmentation correctness

The ICU segmenter is built not as a batteries included segmenter. We should verify correctness for our use, as I'm suspicous that it did not actually include all of hte locales that we care about. For the fxtranslate-cli implementaiton in Rust, this would be an issue.

For the JS+wasm package, it would be preferable to use the Intl.Segmenter through something like js-sys, where we can power the segmentation through the fully packaged Intl API, and not have to include and maintain our own.
