from dotenv import load_dotenv

from reviewproof.eval.cli import main

if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
