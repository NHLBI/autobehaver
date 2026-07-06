"""Module entrypoint for mmgen.

Allows running master matrix generation via:

    python -m autobehaver.postinf.mmgen ..."""

from .cli import main

if __name__ == '__main__':
    main()
