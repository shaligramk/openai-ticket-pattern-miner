# openai-ticket-pattern-miner

Cluster customer-support tickets to surface **systemic** bugs — moving support from reactive ticket-closing to proactive engineering action.

> *"Which cluster of tickets, if we fix the underlying issue, would deflect the most volume next week?"*

## What it does

Given a corpus of support tickets, this tool:

1. **Embeds** each ticket with `text-embedding-3-small`.
2. **Clusters** them with HDBSCAN (handles unknown cluster count + noise — better than k-means when you don't know how many bug categories you have).
3. **Summarizes** each cluster with an LLM: root-cause hypothesis, suggested engineering action, exemplar ticket, and a severity score (volume × frustration sentiment × week-over-week growth).
4. **Reports** the result as a static HTML page suitable for GitHub Pages.

## Quickstart

```bash
git clone https://github.com/shaligramk/openai-ticket-pattern-miner.git
cd openai-ticket-pattern-miner
cp .env.example .env       # paste your OPENAI_API_KEY
make install
make demo                  # generate → embed → cluster → summarize → report
open reports/report.html
```

End-to-end runtime on the bundled 500-ticket synthetic dataset: ~2 minutes.

## Why this exists

Most "AI for support" projects are reactive — answer one ticket faster. The harder, more valuable problem is **finding the bug behind the bug**: spotting when fifty different-looking tickets are actually the same root cause that engineering should fix once.

Built as a portfolio piece for the OpenAI **AI Support Engineer** role, where the work is half ticket-resolution and half "drive initiatives that reduce bugs, improve features."

## Pipeline

| Stage | Module | Output |
|---|---|---|
| Generate synthetic tickets | `ticket_miner.generate` | `data/tickets.jsonl` |
| Embed | `ticket_miner.embed` | `data/embeddings.npz` |
| Cluster | `ticket_miner.cluster` | `data/clusters.json` |
| Summarize | `ticket_miner.summarize` | `data/summaries.json` |
| Report | `ticket_miner.report` | `reports/report.html` |

## Project status

- [x] Project scaffold
- [ ] Synthetic ticket dataset (~500 tickets across ~10 archetypes)
- [ ] Embedding + clustering pipeline
- [ ] Cluster summarization with severity scoring
- [ ] HTML report
- [ ] GitHub Pages deploy
- [ ] Write-up

## License

MIT
