# agent-stability

Growth-ratio monitor for multi-turn LLM agent loops. It judges the prompt before the model call, stops a run whose turns keep climbing, and writes a failure report from that trajectory alone. No extra model call.

This is an independent package. The growth-ratio idea, *Empirical Lyapunov Stability* (2026, [doi:10.5281/zenodo.20722987](https://doi.org/10.5281/zenodo.20722987)).

## Install

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
python -m agent_stability --help
python -m agent_stability --max-turns 10 --safety-k 0.90
python -m agent_stability --corpus ./examples --safety-k 0.92
```

## License

MIT. See [LICENSE](LICENSE).
