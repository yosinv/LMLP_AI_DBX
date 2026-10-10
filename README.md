# Accessible Stays Finder · Databricks AI agent

An AI agent on Databricks that finds **wheelchair-accessible places to stay**. It doesn't trust listing checkboxes: it searches what real guests wrote in their reviews.

```
user: "New York"
        │
        ▼
 Spark SQL over Airbnb + Booking.com datasets  ──►  listings with 100+ reviews
        │                                           where a review mentions "wheelchair"
        ▼
 LangGraph ReAct agent + MCP tools (Nimble)    ──►  Google Maps link per property
        │
        ▼
 Llama 3.3 70B (Databricks Model Serving)      ──►  ranked recommendations with
                                                    the accessibility evidence quoted
```

## What's in here

| File | What it does |
|---|---|
| `LMLP_AI_DBX.py` | The main pipeline: review-based accessibility search, MCP-powered map enrichment, LLM summary. Traced with MLflow autologging. |
| `ai agent.py` | Experiments: a LangChain SQL agent over the same data, and **CLIP** zero-shot detection of amenities (kitchen, elevator, parking and more) in listing photos. |

## Stack

`Databricks` · `Spark SQL` · `Databricks Model Serving (Llama 3.3 70B)` · `LangGraph` · `MCP` · `MLflow` · `CLIP` · `Python`

## Design notes

- **Reviews over metadata.** Hosts' accessibility tags are often missing or wrong. Guest reviews mentioning wheelchairs are a stronger signal, so the SQL filters on review text inside JSON arrays (`EXISTS(FROM_JSON(...))`).
- **The LLM never writes SQL here.** The agent's input reaches Spark only as a bound parameter (`:location_pattern`), never by string formatting, so a prompt-injected location can't change the query.
- **Short-lived credentials.** The Databricks token is minted per run with a 20-minute lifetime instead of a long-lived PAT.

## Run it

Runs as a Databricks notebook (it uses the notebook's `spark` session and top-level `await`).

1. Copy `.env.example` to `.env` and set `NIMBLE_API_KEY`.
2. Make sure your workspace has access to the `databricks-meta-llama-3-3-70b-instruct` serving endpoint and the Bright Initiative datasets.
3. Run `LMLP_AI_DBX.py` in a notebook attached to a cluster.

## Author

Yosi Navaro · Data Engineering Manager · [github.com/yosinv](https://github.com/yosinv)
