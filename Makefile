build:
	python -m venv .venv
	/bin/bash -c "source .venv/bin/activate && pip install --upgrade pip && pip install -r dev.txt"

install:
	brew install graphviz


clean:
	rm -rf .venv