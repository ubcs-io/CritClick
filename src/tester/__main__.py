"""Allow ``python -m tester`` as an entry point.

Simply delegates to ``tester.cli.main()``.
"""

from .cli import main

main()
