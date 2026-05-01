PYTHON := python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: install demo generate embed cluster summarize report test clean

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -e ".[dev]"

generate:
	$(PY) -m ticket_miner.generate --n 500 --out data/tickets.jsonl

embed:
	$(PY) -m ticket_miner.embed --in data/tickets.jsonl --out data/embeddings.npz

cluster:
	$(PY) -m ticket_miner.cluster --in data/embeddings.npz --out data/clusters.json

summarize:
	$(PY) -m ticket_miner.summarize --tickets data/tickets.jsonl --clusters data/clusters.json --out data/summaries.json

report:
	$(PY) -m ticket_miner.report --tickets data/tickets.jsonl --clusters data/clusters.json --summaries data/summaries.json --out reports/report.html

demo: generate embed cluster summarize report
	@echo ""
	@echo "Demo complete. Open reports/report.html"

test:
	$(PY) -m pytest -q

clean:
	rm -f data/embeddings.npz data/clusters.json data/summaries.json reports/report.html
