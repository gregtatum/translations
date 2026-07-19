"""``python -m fxtranslate`` entry point — defers to the CLI ``main`` (the same
function bound to the ``fxtranslate`` console script). The conformance harness
invokes the CLI this way so it works from the editable/wheel install with no PATH
assumptions."""

from .cli import main

main()
