# Accessible Travel AI Agent with Databricks + MCP

An experimental **AI agent for discovering accommodation listings with accessibility-related review signals**. The implementation combines Databricks SQL data, a Databricks-hosted LLM and MCP-powered map-link lookup.

> **Important:** A review mentioning “wheelchair” is only a signal, **not proof that accommodation is accessible**. Always verify accessibility requirements directly with the property before booking.

## Architecture

```text
Travel destination
       ↓
Databricks SQL: Airbnb + hotel listing datasets
       ↓
Filter reviews for accessibility-related terms
       ↓
MCP-powered agent finds map links
       ↓
Databricks LLM summarizes evidence and caveats
       ↓
Travel recommendations with source context
```

## Tech stack

Python, Pandas, Spark SQL, Databricks SDK, `langchain-databricks`, LangGraph ReAct agent, MCP client adapters, MLflow tracing and `python-dotenv`.

## Source

- `LMLP_AI_DBX.py` — notebook-oriented proof of concept that queries two Databricks datasets and summarizes accommodation listings.
- Requires an existing Databricks workspace, access to the referenced tables, an appropriate model-serving endpoint, a Spark session and a configured MCP provider.
- The sample runs in a notebook-style environment; its final top-level `await` is **not valid in a standard Python script**.

## Configuration and security

1. Use Databricks authentication through the supported SDK configuration; never store long-lived tokens in source control.
2. Set `NIMBLE_API_KEY` as a secret/environment variable available to the execution environment.
3. Ensure the Databricks user has least-privilege read access to the listed datasets.
4. Confirm the available model endpoint and MCP transport for your environment.
5. Do not use actual guest personal information or restricted datasets for public demos.

**Security notice:** An API credential was previously embedded in a public revision. It must be **revoked/rotated** by its owner; updating the working tree does not erase historical Git commits.

## Current limitations

- The SQL query interpolates a destination string directly. Before production use, validate inputs and use parameterized queries where supported.
- The code depends on a notebook Spark context and environment-specific Databricks datasets.
- Matching review keywords does not verify property accessibility.
- No reproducible public dataset or automated tests are provided yet.
- MCP map URLs and LLM summaries should be treated as unverified suggestions.

## Example (illustrative, not tested)

**Destination:** New York  
**Goal:** Find hotels or vacation rentals whose reviews mention wheelchair access, then summarize the available evidence and provide map links for follow-up verification.

## Next milestones

- [ ] Rotate exposed credential and purge it from Git history where feasible
- [ ] Replace interpolated SQL with validated/parameterized inputs
- [ ] Split notebook code into testable modules and a CLI
- [ ] Add synthetic demo data and offline tests
- [ ] Add MLflow evaluation and trace inspection examples
- [ ] Introduce `pyproject.toml`, `.gitignore`, CI and screenshots

**Status:** Databricks-focused experimental prototype; not a booking or accessibility verification service.
