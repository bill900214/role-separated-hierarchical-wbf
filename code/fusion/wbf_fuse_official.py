#!/usr/bin/env python3
"""Compatibility entry point for the renamed portable WBF helper.

The filename is retained only to avoid breaking old notes. The implementation
is not the original MMDetection WBF used by the reported Level-II/III runs.
"""

from wbf_portable_helper import main


if __name__ == "__main__":
    main()
