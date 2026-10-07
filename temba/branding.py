"""Terminal branding for TEMBA: the banner shown by the installer and the CLI.

Colours follow the project palette (slate navy, rose, teal, amber). They are
switched off when output is not a terminal or NO_COLOR is set.
"""
import os
import sys

from . import __version__, EXPANSION, TAGLINE

# 24-bit colours of the project palette
NAVY, ROSE, TEAL, AMBER, GREY = (46, 64, 87), (209, 73, 91), (0, 121, 140), (237, 174, 73), (141, 150, 163)

LOGO = r"""
 ████████╗ ███████╗ ███╗   ███╗ ██████╗   █████╗
 ╚══██╔══╝ ██╔════╝ ████╗ ████║ ██╔══██╗ ██╔══██╗
    ██║    █████╗   ██╔████╔██║ ██████╔╝ ███████║
    ██║    ██╔══╝   ██║╚██╔╝██║ ██╔══██╗ ██╔══██║
    ██║    ███████╗ ██║ ╚═╝ ██║ ██████╔╝ ██║  ██║
    ╚═╝    ╚══════╝ ╚═╝     ╚═╝ ╚═════╝  ╚═╝  ╚═╝
"""

# a small spectrum: the double horn of a rotating disc, drawn in block heights
SPECTRUM = "▁▁▁▂▅█▇▆▆▇█▅▂▁▁▁"


def _colour_ok(stream=None):
    stream = stream or sys.stdout
    return (hasattr(stream, "isatty") and stream.isatty()
            and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb")


def paint(text, rgb, bold=False, stream=None):
    if not _colour_ok(stream):
        return text
    r, g, b = rgb
    return f"\033[{'1;' if bold else ''}38;2;{r};{g};{b}m{text}\033[0m"


def banner(stream=None, compact=False):
    """The TEMBA banner: wordmark, expansion, version."""
    stream = stream or sys.stdout
    if compact:
        line = (paint("TEMBA", NAVY, True, stream) + " " + paint(SPECTRUM, TEAL, stream=stream)
                + "  " + paint(f"v{__version__}", GREY, stream=stream))
        print(line, file=stream)
        return
    rows = LOGO.strip("\n").splitlines()
    # the wordmark shades from navy (top) to teal (bottom)
    for i, row in enumerate(rows):
        f = i / max(len(rows) - 1, 1)
        rgb = tuple(int(NAVY[k] + f * (TEAL[k] - NAVY[k])) for k in range(3))
        print(paint(row, rgb, True, stream), file=stream)
    print(paint("   " + SPECTRUM + "   " + SPECTRUM[::-1], AMBER, stream=stream), file=stream)
    print(paint(f"   {EXPANSION}", NAVY, True, stream), file=stream)
    print(paint(f"   {TAGLINE}" + " " * 8 + f"v{__version__}", GREY, stream=stream), file=stream)
    print(file=stream)


def ok(msg, stream=None):
    print(paint("  ✓ ", TEAL, True, stream) + msg, file=stream or sys.stdout)


def warn(msg, stream=None):
    print(paint("  ! ", AMBER, True, stream) + msg, file=stream or sys.stdout)


def fail(msg, stream=None):
    print(paint("  ✗ ", ROSE, True, stream) + msg, file=stream or sys.stdout)


def heading(msg, stream=None):
    print(paint(msg, NAVY, True, stream), file=stream or sys.stdout)


if __name__ == "__main__":
    banner()
