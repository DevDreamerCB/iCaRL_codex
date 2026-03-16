# AutoResearch Usage

## Files

- executor: `autoresearch_icarl.py`
- executor wrapper: `run_autoresearch.sh`
- agent controller: `agent_loop.py`
- agent wrapper: `run_agent_loop.sh`
- control helper: `agentctl.sh`
- program: `research/program.md`
- state: `research/state.json`
- long-term memory: `research/MEMORY.md`
- daily memory: `research/daily_YYYYMMDD.md`
- executor log: `metrics/autoresearch.log`
- agent log: `metrics/agent_loop.log`
- event stream: `metrics/events.jsonl`
- agent status page: `metrics/agent_status.md`

## Recommended Way To Keep It Running

This machine does not have `tmux`, so use `nohup` through the wrapper scripts.

### Better Option: Agent + Executor

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
./agentctl.sh start 12
```

This starts the two-layer loop:

1. `agent_loop.py` reads state and recent results
2. it processes new experiment events
3. it updates memory files
4. it launches `autoresearch_icarl.py` for one research cycle at a time
5. if progress stalls, it switches research direction automatically

### Simpler Option: Executor Only

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
./agentctl.sh start-runner 12
```

This runs the lower-level research executor only.

## Check Progress

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
./agentctl.sh status
tail -f metrics/agent_loop.log
tail -f metrics/autoresearch.log
```

## Stop

```bash
cd /data1/bochen/cbcontinual/iCaRL_codex
./agentctl.sh stop
./agentctl.sh stop-runner
```

If the pid file is stale or you want to inspect raw files:

```bash
rm -f metrics/autoresearch.pid
rm -f metrics/agent_loop.pid
cat metrics/agent_status.md
tail -n 20 metrics/experiments.csv
cat metrics/latest.md
```

## What The Two-Layer System Does

### Agent Controller

1. Reads `research/state.json`
2. Processes new events from `metrics/events.jsonl`
3. Updates `research/MEMORY.md` and the daily memory log
4. Checks best confirmed result and stale rounds
5. Chooses the current research direction
6. Launches one executor cycle
7. If several nearby candidates fail, switches direction automatically

The controller also updates:

- `metrics/agent_status.md`
- per-family acceptance/rejection memory in `research/state.json`
- `research/MEMORY.md`

### Executor

1. Reads current state and metrics
2. Generates nearby candidates inside the current direction
3. Runs `screen`
4. Promotes promising runs to `confirm`
5. Updates:
   - `metrics/experiments.csv`
   - `metrics/latest.md`
   - `research/results.tsv`
   - `research/notes.md`
   - `research/state.json`
   - `metrics/events.jsonl`
6. Commits and pushes when a confirmed improvement is accepted

## Important Limitation

The chat itself is not the long-running process.

To let the research continue after you disconnect, the thing that must stay alive is:

- `run_agent_loop.sh`

That is the closest local approximation to an always-on online research agent on this machine.
