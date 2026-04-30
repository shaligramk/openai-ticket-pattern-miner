.PHONY: install demo generate embed cluster summarize report test clean

install:
	pip install -e ".[dev]"

generate:
	python -m ticket_miner.generate --n 500 --out data/tickets.jsonl

embed:
	python -m ticket_miner.embed --in data/tickets.jsonl --out data/embeddings.npz

cluster:
	python -m ticket_miner.cluster --in data/embeddings.npz --out data/clusters.json

summarize:
	python -m ticket_miner.summarize --tickets data/tickets.jsonl --clusters data/clusters.json --out data/summaries.json

report:
	python -m ticket_miner.report --tickets data/tickets.jsonl --clusters data/clusters.json --summaries data/summaries.json --out reports/report.html

demo: generate embed cluster summarize report
	@echo ""
	@echo "Demo complete. Open reports/report.html"

test:
	pytest -q

clean:
	rm -f data/embeddings.npz data/clusters.json data/summaries.json reports/report.html
