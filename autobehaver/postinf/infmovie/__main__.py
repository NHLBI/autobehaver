"""Module entrypoint for infmovie.

Allows running inference-overlay movie generation via:

    python -m autobehaver.postinf.infmovie ...
"""

from .cli import main


if __name__ == "__main__":
    main()
