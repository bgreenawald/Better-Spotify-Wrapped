# Slopify Wrapped — Consolidated Code Review

**Last updated:** 2026-10-29 against `690f11c`.

> **Fix progress (2026-10-29, through `2df309f`).** Resolved: **P0 #1–#4**, **P1 #5–#8**, the P1 repo-posture items, and from P2 ingest: `DATA_DIR`, the retry/backoff defects, and the missing `logging.basicConfig`. The `CLAUDE.md` corrections are done too, so it no longer misleads agent sessions. Tests went 7 → 32. Still open and highest-value: **#9** (`get_top_albums`), **#10** (UTC bucketing), **#11** (averages), **#12**/**#13** (ranks and name keying), `slopify status`, and the performance and UX sections.

This is the single, current review doc for the project. It merges:

- the **June 2026 review** (`CLAUDE_FABLE_CODE_REVIEW.md`, reviewed 2026-06-11) — 22 numbered findings, of which **21 are still present**; and
- the **October 2026 review** — five parallel reviews covering dashboard/UX, metrics/SQL, ingest/API, and docs/tooling, plus measurements taken against the live database.

Provenance is marked on every finding so nothing is lost and nothing is double-counted:

| Tag | Meaning |
|---|---|
| **[Jun #n]** | Raised in the June review as finding *n*. Still present unless stated. |
| **[New]** | First raised in October. |
| **[Verified]** | Reproduced by running code against the live DB or CLI — not inferred from reading source. |
| **[Fixed]** | Resolved since June. |

Measurements come from the live `data/db/music.db`: 534 MB, 525,043 plays, 517,288 tracks, 3 users. A traceability table mapping all 22 June findings to their location in this document is at the end.

**Nothing in the repo has been changed by either review.**

---

## TL;DR

The project is structurally sound — clean module boundaries, real features (sunburst genre hierarchy, Venn social overlap, album MQPC scoring), excellent docstrings, ruff clean, 27 files formatted. What it lacks is a **working path from zero to a dashboard**, and **correct numbers in the charts it already draws**.

Two things outrank everything else, and neither was in the June review:

1. **The project cannot be set up from scratch.** `slopify init-db` — the first command in the README — crashes, and the artist-genre pipeline the README documents is unreachable from the CLI.
2. **The dashboard shows wrong numbers.** The last day of every date range is silently dropped, and genre percentages sum to 244%.

Separately, and unrelated to code quality: the repo is **public with no license** (all rights reserved by default), and `SEED.sql` ships **four real family members' names**. Both are one-line fixes worth doing today.

The June review's priorities remain valid and are folded in below. Its ordering is superseded, because measurement showed two of its performance concerns are minor and one of its "hazards" is in fact a live crash.

---

## P0 — Broken for a new user

### 1. `slopify init-db` crashes on a fresh database — **[Jun #13, Verified — FIXED `0bb2ab5`]**

June flagged the naive semicolon split as a *hazard* ("will break on any semicolon inside a string literal, comment, or `CREATE MACRO` body"). It is not hypothetical — it is a hard failure today, and it blocks first-time setup entirely:

```
$ slopify init-db --db fresh.db --ddl DDL.sql
duckdb.duckdb.ParserException: Parser Error: syntax error at or near "hierarchy"
$ echo $?
1
```

Root cause: `src/cli.py:35` splits `DDL.sql` on `;`, and `DDL.sql:63` has a semicolon **inside a comment**:

```sql
-- Canonical genres (slug-based external key; hierarchy modeled separately)
```

The split yields `hierarchy modeled separately)` as a bogus statement. Two more comments have the same problem (`DDL.sql:188`, `:191`).

Worse, the statements run outside a transaction, so the crash leaves a **half-created database**. Re-running fails with a *different* error, because `CREATE TABLE` has no `IF NOT EXISTS` (only the indexes and views do):

```
duckdb.duckdb.CatalogException: Catalog Error: Table with name "dim_users" already exists!
```

So the obvious recovery also fails. `cli.py:68`'s docstring claims it is "Safe to run on existing databases" — it is not. `ingest-history --apply-ddl` (README Step 2/3) hits the same bug.

**Fix (verified to work):** hand the whole file to DuckDB, which accepts multi-statement strings:

```python
conn.execute(ddl_path.read_text())
```

This creates all 21 objects correctly. Then add `IF NOT EXISTS` to the 19 `CREATE TABLE`s and wrap the whole thing in one transaction so a failure can't leave a partial schema.

### 2. Nothing in the CLI populates the artist cache — **[New, Verified — FIXED `0bb2ab5`]**

`fetch_artists` and `fetch_albums` (`src/api/api.py:364`, `:405`) are called from exactly two places: `load_api_data` (`api.py:460`) and the `__main__` block (`api.py:1559`). **`load_api_data` has zero callers anywhere** in `src/`, `dashboard/`, or `tests/`, and `src/cli.py` does not import it.

Consequence: none of the 10 CLI commands ever writes to `data/api/cache/artists/`. README Step 5 (`ingest-artist-genres`, which *reads* that cache) therefore has nothing to read on a fresh install — **genres cannot be produced from scratch by the documented flow.** The existing 34,586 cached artists were populated out of band.

This is why "cannot be set up from scratch" is the right framing rather than just "init-db is broken": fixing #1 alone still does not get a new user to a working genre view.

June listed `load_api_data` and the `__main__` block as **dead code** (#7). They are — but the live `fetch_artists`/`fetch_albums` hang off them, so deleting them naively would remove the only code that can populate the artist cache. Wire the fetch into the CLI *before* deleting.

### 3. One bad track ID discards up to 49 good ones, every run — **[Jun #1 — FIXED `0bb2ab5`]**

Spotify's `/tracks` endpoint returns `null` for IDs it cannot resolve. `_save_cache` does `item["id"]` (`api.py:143`) → `TypeError` → caught by the blanket `except Exception` at `api.py:359` → **the entire 50-item batch is discarded**, including items already fetched successfully. Saving also stops at the first `None`, so later items in the batch are never cached.

There is no negative cache, so the same batches fail on every subsequent run and those DB columns stay `NULL` forever. A single malformed ID produces a 400, which `_fetch_with_retry` re-raises at `api.py:310` and loses the batch the same way.

Fix: filter `None` before caching, log the specific null IDs, record dead IDs so they are not re-selected, and bisect the batch on a 400. Same defect in the artist and album variants (`api.py:395`, `:436`).

### 4. Cache writes are not atomic, and corruption is permanent — **[Jun #2 — FIXED `0bb2ab5`; the `CLAUDE.md` file-locking claim still needs correcting]**

`_save_cache` (`api.py:142-146`) is a plain `open(path, "w")` + `json.dump`. A crash or Ctrl-C mid-write leaves truncated JSON. `_get_cached_ids` counts any `*.json` as a cache hit, so the next run calls `_load_cache` in a list comprehension (`api.py:335`) and dies with `JSONDecodeError` — and **never self-heals**; the user must find and delete the file by hand.

June noted this is safe *today* only because `_save_cache` is called from the main thread in the `as_completed` loop. Two concurrent CLI invocations sharing the cache dir will still race.

`CLAUDE.md:120` and `:175` both claim "Thread-safe caching with file locks." There is **no file locking anywhere** in the codebase — no `filelock` dependency, no `fcntl`, only an in-process `threading.Lock` at `api.py:96`. The docs describe behavior that does not exist.

Fix: write to a temp file in the same directory and `os.replace`; treat a decode error in the loader as a cache miss and refetch. Then correct `CLAUDE.md`.

---

## P1 — The dashboard shows wrong numbers

### 5. The last day of every date range is silently dropped — **[New, Verified — FIXED `0bb2ab5`]**

```
Dec 31 2024 -- in DB: 45, returned by get_filtered_plays: 0, DROPPED: 45
max ts returned for "2024": 2024-12-30 22:26:36
```

Callers pass a date-only / midnight end (`callbacks.py:586` uses `datetime(year,12,31)`), and `preprocessing.py:275` filters `played_at <= ?`. Midnight excludes the whole day. This affects the Wrapped year view, the Trends range (`callbacks.py:~1797`), and the Social range (`social.py:259`).

It has stayed invisible because `exclude_december` defaults to **Yes**, masking it on the Wrapped tab specifically — but it silently truncates every custom range on the other tabs.

Fix: `played_at < end + INTERVAL 1 DAY`, or normalize the end to `23:59:59.999999`.

### 6. Genre percentages sum to 244% — **[New, Verified — FIXED `0eec808`]**

```
genre rows: 786
sum(play_count) = 83,756   <- for 34,343 actual plays (2.44x)
sum(percentage)  = 243.6%
```

`get_top_artist_genres` counts each play **once per genre tag on its artist**, then divides by total plays. An artist tagged with 3 genres contributes 3 plays. So "Pop — 18%" on the dashboard is not a share of anything; numerator and denominator come from different universes.

Fix (pick one, and label it in the UI): present as "share of genre-tagged plays" using the genre-sum denominator, or distribute each play's weight fractionally across its artist's genres. The schema was *designed* for the second option — `track_genres.score` and `agg_*.avg_score_weighted` exist for exactly this — but see #8.

### 7. Filters are inconsistent between tabs — **[Jun #5 (partial), New (the drift) — FIXED `2df309f`; the single-SQL-builder refactor is still open]**

June noted the base-plays filter is implemented three times. The new finding is that the three copies have **drifted**, so the same conceptual filter now behaves differently per tab:

| | `get_filtered_plays` (Wrapped/Trends) | `social._base_plays_sql` | `filter_songs` (legacy) |
|---|---|---|---|
| filters `skipped` | **no** | yes | yes |
| name exclusions | case-**sensitive** | `LOWER()` | — |
| genre exclusion | `track_genres` + `artist_genres` | `artist_genres` only | — |

So Wrapped includes skipped plays that Social excludes, and an exclusion typed in different case works on one tab and not another. Collapse to one SQL builder.

### 8. Half the schema is dead; the rollups are never populated — **[New, Verified — FIXED `aa8e5df`]**

> Resolved by deletion, per table. Removed from `DDL.sql`: the mood half (`dim_moods`, `map_mood`, `track_moods`) — no code and no data source, since Spotify audio features are never fetched; `track_genres` — read by two guarded fallbacks, never written, so neither could fire; and the four `agg_*` rollups — no writer, no reader, and a monthly aggregate off `fact_plays` measures ~4ms. The 74-line dead `track_genres` fallback in `metrics.py` and the dead branch in `preprocessing.py` are gone, the stale empty `track_artists` was dropped from the live DB, and `DDL.sql` now declares exactly the 13 objects the live database has. A test asserts that set so the schema cannot drift back.

`DDL.sql` declares 21 objects. The live DB has 14. Missing entirely: `dim_moods`, `map_mood`, `track_moods`, `track_genres`, and all four `agg_user_{genre,mood}_{daily,monthly}` tables.

Grepping `src/`, `dashboard/`, and `tests/`:

- The **entire mood half** of the schema (`docs/TASK.md`'s goal #2) is referenced by **zero lines of code**. It is aspiration, not implementation.
- The four `agg_*` rollup tables have **no writer and no reader**. Every trend is recomputed per request. `agg_user_genre_monthly` would serve `get_genre_trends` directly.
- `track_genres` is **read** (`preprocessing.py:304`, `metrics.py:808`) but **never written** by any code. Those reads sit behind `_table_exists` guards and a bare `except: pass` (`metrics.py:867` — June #3's showcase example), so the fallback silently never fires and you would never know.
- `DDL.sql:81` names MusicBrainz, Last.fm, Discogs and TheAudioDB as tag sources. **No code uses any of them.**
- The live DB additionally has a stale empty `track_artists` table (0 rows) that `DDL.sql` does not declare.

Decide per table: populate it, or delete it from `DDL.sql`. Right now `DDL.sql`, `CLAUDE.md`, and `TASK.md` all described a mood/rollup system that did not exist — actively misleading, including to future agents reading `CLAUDE.md`.

### 9. `get_top_albums` credits plays to the wrong albums — **[Jun #10 (threshold only), New (the rest)]**

Plays are matched on `(track_name, primary_artist)` (`metrics.py:368`), so **every** album containing a same-named track gets full credit — compilations, deluxe editions, Greatest Hits. The function's comment (`metrics.py:299`) promises singles are summed with the album version, but `albums_in_scope` (`:332`) is built only from albums of *played* track IDs, so a single's plays are credited only if a different track on that album was also played.

Further defects in the same function:

- **Fuzzy dedup runs at `similarity_threshold=70.0`** (`metrics.py:526`) against a documented default of 90 — June #10 flagged the mismatch; the new finding is why 70 is actively wrong. `fuzz.token_set_ratio` returns **100** whenever one name's tokens are a subset of the other's, so "Love" / "Love Songs" and "Live" / "Live in Paris" merge. Losing albums' plays are **dropped, not summed** (`metrics.py:198`).
- The Python dedup sums `play_count` across duplicate rows (`metrics.py:452`); two track IDs with the same name each carry the full count → double counting.
- `HAVING COUNT(*) > 5` (`:401`) runs *before* the dedup, so albums can fall under the threshold and still display.
- The album score collapses to 0 whenever under ~half an album was played (MQPC is the mean of the middle 50%), so many albums tie at 0. With `threshold=0`, `t_eff` equals the total track count. Heavily played 30-track deluxe editions are also penalized. The middle slice is not exactly 50% (for n=8 it is 5 items).
- `pwa.primary_artist = tpa.primary_artist` is NULL-unsafe → tracks with no artist silently get 0 plays.
- `WHERE rn = 1 OR rn IS NULL` (`:322`, `:357`) is dead logic — the window is over a LEFT JOIN, so `rn` is never NULL.

Consider stripping edition suffixes (`(Deluxe)`, `(Remastered)`, `(Taylor's Version)`) and matching on `fuzz.ratio`, and extracting the MQPC math into a pure, testable function.

### 10. Everything is bucketed in UTC — **[New]**

`played_at` is naive UTC (`db_ingest.py:70`). All day/month bucketing, "Most Active Day", `unique_days`, and `exclude_december` are UTC. Late-evening US listening lands on the next day; plays near a month boundary land in the wrong month. Nothing converts to local time. Either store a local timestamp or convert at query time, and say which in the UI.

### 11. Averages use three different denominators — **[Jun #15 + New]**

- `trends.py:94` divides by **days in the calendar month**, so first/current partial months always read low. *(June #15.)*
- Wrapped's `avg_daily_hours` (`callbacks.py:~650`) divides by **days with any listening**, which overstates. *(New.)*
- Two "average per day" figures on the same dashboard therefore mean different things.

Also: `get_most_played_tracks` returns `percentage` as a 0–1 fraction (`metrics.py:80`) while artists, genres and trends return 0–100.

### 12. Rank and tie semantics are inconsistent — **[New]**

- `DENSE_RANK() OVER (… ORDER BY play_count DESC, genre)` (`trends.py:216`, `:463`) uses a unique tiebreaker, so it behaves like `ROW_NUMBER` and ties never share a rank.
- Artist trends use a pandas dense rank with true ties (`trends.py:371`).
- Social ranks use `DENSE_RANK` with an `entity_name` tiebreaker (`social.py:145`, `:167`, `:197`), so `joint_rank` (a mean of ranks) is arbitrary among ties.
- "Top" lists `ORDER BY play_count DESC, name` + `LIMIT` silently cut ties at the boundary (`metrics.py:82`, `:940`).
- `get_playcount_by_day(top_only=True)` breaks ties alphabetically with no indication a tie occurred.

### 13. Entities are keyed by display name, not ID — **[New]**

Tracks, artists and `unique_tracks` are grouped by name (`metrics.py:68`, `:931`; `trends.py:80`, `:310`). Two different songs named "Intro" by one artist merge; renamed artists split. Social ranks by `track_id` (`social.py:138`), so the same song released as a single and as an album track is **two** entities in the social view and **one** in Wrapped.

### 14. Other statistical quality issues — **[New]**

- **Missing months are not zero-filled.** Artist, track and genre trends emit only months with plays, so line charts interpolate over silent months.
- **Social "totals" are capped.** `social.py:586` caps every Venn total at `CONSIDER_LIMITS` (250/100/50), so a pair sharing 800 tracks displays 250. June #22 asked for a comment; the display is misleading regardless.
- **No diversity or concentration measure exists** — no entropy, Gini, HHI, top-N share, or repeat rate — despite `docs/TASK.md` calling for entropy. Shannon entropy is cheap: `-SUM(p*ln p)` over per-artist play share.
- **Play counts are unweighted by duration.** A 31-second play counts the same as a full one, while `avg_duration_min` is shown alongside; the mean includes truncated plays, and no completion rate is computed.
- **No minimum-plays threshold**, so a month with 3 plays can rank a genre #1.
- **The album score is opaque** — the constants (0.5, 15, 0.3, `>5` tracks) are undocumented in the UI, with no small-sample confidence.
- **Latent play duplication.** `preprocessing.py:364` and `social.py:110` do `LEFT JOIN bridge … role='primary'` with no dedupe, so a track with two primary rows duplicates every play. Ingest writes only index 0 as primary (`api.py:887`), so this does not happen today, but the DDL PK permits it — and `v_primary_artist_per_track` exists for exactly this case and is unused here.

---

## P1 — Repo posture — **[New, Verified]**

Two items with consequences outside the code.

### Public repo, no license

`gh repo view` reports `"visibility": "PUBLIC"` and `"licenseInfo": null`, and there is no `LICENSE` file. By default that means **all rights reserved** — nobody can legally reuse it, which is probably not the intent of a public project. Add a license (MIT if you want it freely usable).

Relatedly: `dashboard/assets/highcharts.js` (v11.2.0, ~280 KB) and five other Highcharts modules are **vendored into a public repo**, and Highcharts' free license is non-commercial. `pyproject.toml:46` acknowledges this for the dev dependency, but vendoring the bundle publicly is a different posture than using it locally. Worth a deliberate decision — and note `highcharts-core` (the Python dev dep) is imported by **no Python anywhere**, so it can simply go.

### Personal data in a tracked file

`SEED.sql` is tracked and public, and contains four real full names:

```sql
INSERT INTO dim_users (user_id, display_name, country) VALUES
('ben.greenawald', 'Ben Greenawald', 'US'),
('margo.greenawald', 'Margo Greenawald', 'US'),
...
```

These are family members, not only you, and the repo is public. Replace with generic placeholders (`user_one` / `Example User`) and keep the real seed local.

The file is also **broken as a seed**: it re-declares `CREATE TABLE dim_users` with no `IF NOT EXISTS`, so running it after `DDL.sql` fails.

---

## P2 — Correctness and robustness of ingest

- **`NaT` becomes the string `'NaT'`** — **[New]**. `io.py:35` coerces bad timestamps to `NaT`; `_to_timestamp(pd.NaT)` returns the *string* `'NaT'`, which passes the `.notna()` guard and reaches the timestamp insert (`db_ingest.py:384`), failing the single transaction and rolling back the whole ingest with an unhelpful message. A missing `spotify_track_uri` (NaN) crashes `_extract_spotify_track_id` with `AttributeError` (`db_ingest.py:46`, no type guard). Check `pd.isna` first and report a dropped-row count.
- **`io.py` robustness** — **[Jun #16]**. No `utc=True` on `pd.to_datetime` (works today only because Spotify exports are uniformly `Z`-suffixed; mixed offsets would yield object dtype and `.dt.tz_localize(None)` would raise — `db_ingest._to_timestamp` already does this correctly). An empty or nonexistent directory yields an empty frame with no columns and no error. No guard that a JSON file contains a list of records. **[New]** on top: it globs every `*.json` including `Streaming_History_Video_*` files from the same export, unsorted, with no filename in error messages.
- **`DATA_DIR` is required for DB-only commands** — **[New]**. The `SpotifyDataCollector` constructor raises `ValueError` if unset (`api.py:78`) and creates the directory — but `DATA_DIR` is only *used* by dead code (`load_api_data`, `__main__`). The README's env section lists only the two Spotify keys, so a new user's `ingest-track-metadata` fails on a variable they were never told to set.
- **`populate_track_artists` has no transaction and drops indexes** — **[Jun #11 + New]**. June flagged the whole-table scans into Python sets (`api.py:914-921`: existing artists, all track IDs, and the entire `bridge_track_artists` table) while every other populate function uses SQL anti-joins. **[New]**: there is also no `BEGIN`/`COMMIT` (`api.py:911-984`), and indexes are dropped (`:949`) and recreated **only on the success path** — so a mid-insert failure leaves the database without indexes and does not tell the user. Recreate in `finally`. June also asked whether the DuckDB ART workaround is still needed on ≥1.3; worth re-testing and removing if stale.
- **Retry/backoff defects** — **[Jun #9 + New]**. A 429 consumes one of the 3 `max_retries` attempts *and* sleeps `Retry-After`, so three 429s exhaust the budget even though the circuit breaker implies you want to continue after cooldown. The adaptive delay only ever increases (`api.py:235`) and never decays, so one bad patch leaves the collector at up to 5 s/request for the rest of the process. `_check_request_budget` (`api.py:168`) is a soft cap under `max_workers > 1`. **[New]**: the **rate-limit lock is held while sleeping** (`api.py:162`), serializing every worker — so `max_workers > 1` buys nothing; three 429s trigger a 300 s sleep and *then* the batch is lost anyway at `:318`; `int(e.headers.get("Retry-After"))` is unbounded and crashes on a missing header; and the `except Exception` at `:311` retries non-API errors including the `TypeError` from #3. *(Note: June's claim that the terminal `raise SpotifyException(429, …)` misreports non-429 exhaustion does not hold — non-429 `SpotifyException`s re-raise immediately — but the 429 labelling is still misleading.)*
- **Zero-ingest looks like success** — **[New]**. If every batch fails (bad credentials, no network), the populate functions return 0 and the CLI exits **0** with "Updated rows: 0". Credentials are validated lazily, so a wrong secret surfaces as a per-batch `logging.error` rather than one actionable message up front.
- **`logging.basicConfig` is never called** — **[New]**. Every `logging.info` in the codebase is discarded and warnings print as bare lines. There is no `--verbose`. This compounds June #3: even the handlers that *do* log are invisible.
- **Dedupe key granularity** — **[Jun #18]**. Plays are deduped on `(user_id, track_id, played_at)` at second resolution (`db_ingest.py:~261` uses `timespec="seconds"`; anti-join at `:378`). Two legitimate plays of one track inside one second merge — effectively impossible in practice, but `ts` is the lynchpin across re-ingests of overlapping export files and deserves a comment. The idempotency claim at `README.md:69` is **accurate**.
- **`offline_timestamp` is always NULL** — **[New]**. The export supplies an epoch int, `_to_timestamp` ignores ints, and the column is dropped at `db_ingest.py:332`.
- **Stale comments** — **[New]**. `api.py:555` and `:1049` reference a UNIQUE constraint on `track_isrc` that `DDL.sql` does not have. ISRC dedupe drops legitimate duplicates, so those tracks stay NULL and are re-selected every run.
- **`load_dotenv(override=True)` is unconditional** in `api.py:20` and `api/utils.py:11`, so CLI runs always clobber shell env vars — while `dashboard/application.py:37` only overrides when `ENVIRONMENT` is set. `CLAUDE.md:166` documents the gated behavior as universal. **[New]**
- **Foreign keys are unchecked** — **[New]**. `ingest-track-artists` filters bridge rows to existing tracks and silently drops the rest.

---

## P2 — Performance (measured first)

June raised two performance concerns (#12, #14). Measurement shows one is the single biggest win available and the other is negligible — so the ordering below supersedes June's.

| Operation | Measured |
|---|---|
| Full 525k-row `v_plays_enriched` scan → pandas | **0.67 s** |
| Monthly aggregate over `fact_plays` | **0.004 s** |
| Dashboard import + `create_app()` | **2.4 s** |
| Fresh read-only DuckDB connection, open+close | **4.5 ms** (×15/interaction ≈ **68 ms**) |
| Shared-connection cursor | **0.04 ms** |

**DuckDB is not the bottleneck, and neither is connection churn.** **[Jun #14]** June flagged the ~15 per-callback `get_db_connection()` calls (`callbacks.py:227, 259, 291, …`) as worth fixing for latency. At **68 ms per interaction** that is **low priority** — fix it for tidiness (a module-level connection plus `.cursor()`, or a `@contextmanager`), not for speed. Several of those sites also fail to close in a `finally` and reference an unbound `con` if `get_db_connection` raises (`callbacks.py:1305`, `:1792-1856`) — that is a correctness reason to do it.

The real costs are at process and network boundaries:

- **318 MB resident** — **[Jun #12, quantified]**. `application.py:154` loads the entire enriched plays table into pandas at startup, and `layouts.py` plus the preset callbacks use it only for `user_id` lists and `ts.min()/max()`. Replace with `SELECT user_id, MIN(ts), MAX(ts) FROM fact_plays GROUP BY 1`. **This is the single biggest easy win.**
- **~21 MB of JSON per filter change** — **[Jun #12, quantified]**. `daily_counts` with `top_only=False` is 458,748 rows → 21.5 MB serialized, shipped to the browser inside a single `dcc.Store` (`callbacks.py:661-666`, six DataFrames as nested JSON-in-JSON) and re-parsed with `pd.read_json` by 5–6 renderers on every refresh. **[New]**: because `theme-store` is an Input to each server renderer, **flipping dark mode re-runs all of it**. Fix: drop `theme_data` from the server Inputs (the clientside renderer already calls `applyTheme`), and move the payload to a server-side cache keyed by the filter hash, storing only the key in `dcc.Store`.
- **7.9 GB / 517,278 files** in `data/api/cache` — **[New]**. One pretty-printed (`indent=4`) JSON file per ID. `_get_cached_ids` globs all 517k files on **every call**, and `fetch_tracks` returns every cached dict for every requested ID (`api.py:335`), holding gigabytes in memory per command. Compact JSON — or better, a single DuckDB/Parquet cache table — would shrink this by an order of magnitude and give atomicity (#4) and negative caching (#3) for free.
- **A regex in a join condition defeats `idx_artist_genres_artist`** — **[New]**: `REGEXP_REPLACE(TRIM(ag.artist_id), '.*:', '')` (`metrics.py:966`, `:974`; `preprocessing.py:359`, `:380`) runs per row and blocks the index. Normalize the ID once at ingest.
- **`v_primary_artist_per_track` is recomputed every call** — **[New]**. An unmaterialized window over the whole bridge table joined to `dim_artists`, recomputed by 5+ queries per interaction. A `primary_artist_id` column on `dim_tracks` would remove it.
- **Pandas round-trips** — **[New]**. `get_filtered_plays` pulls every play into pandas with a per-row Python `apply` (`preprocessing.py:415`) and `_to_tuple` (`:443`), after which each metric **re-registers a slice back into DuckDB** ~8 times per callback (`callbacks.py:604-639`, `1816-1848`). One SQL query or one temp table would do. `artist_genres_agg` is also built over the **entire** `artist_genres` table on every call (`preprocessing.py:369`), then repeated in `metrics.py:957`.
- **`get_genre_sunburst_rows`** issues 8 CTEs that each re-join the same base, and the fallback re-runs the whole pipeline over a table that is never populated (#8). **`get_artist_trends`** scans the registered relation twice and builds `top_tracks` with `groupby.apply` + `iterrows` (`trends.py:355`) where `STRING_AGG` would do. **`social.py`** re-plans the base query 3 times and does O(N) per-item scans.
- **`SimpleLRUCache` is not thread-safe** — **[Jun #12]**. Dash under gunicorn/waitress with threads can interleave `get`/`set`; a `threading.Lock` is cheap insurance. Its `;`/`|` key separators can also collide (June #22) — hash a `repr` of the tuple instead. Caching covers albums and genres only; the more expensive `filtered` frame is uncached across tabs.

---

## P2 — Dashboard UX

Charts are **Highcharts** rendered clientside (`assets/highcharts_renderer.js`), not Plotly — worth knowing before touching chart code. All **[New]**; the June review did not cover UX.

**Highest value, lowest effort:**

1. **Surprising hidden defaults.** The global filters live in a collapsed hamburger (`layouts.py:101`, `is_open=False`), and both `exclude-december` and `remove-incognito` default to **Yes**, persisted in localStorage. Every chart silently drops December — a very likely source of "my numbers look wrong," and the mask over #5. Add an always-visible active-filter chip row ("Dec excluded · 3 artists excluded"), a **Reset all filters** button (only the date range has one today, `filters.py:241`), and rename the toggles to say *why* ("Exclude December (holiday-music skew)", "Hide private-session plays").
2. **No empty or error states.** Renderers return `None`/`[]` or raise `PreventUpdate` on missing data, so the chart area either goes blank or keeps showing the **stale previous chart**. Add an `_empty_figure(msg)` helper (Highcharts `lang.noData` plus the `no-data-to-display` module) and an `html.Div(role="status")` live region.
3. **Series colors are seven near-identical greens** (`highcharts_renderer.js:91`: `#1DB954 #1ed760 #21e065 #5eb859 #7dd069 #9be082 #b5e8a3`). Multi-series trend charts with 5–15 lines are unreadable, and it is not colorblind-safe. Use a categorical palette (Okabe-Ito or Tableau 10) with Spotify green as series 1, add `dashStyle`/markers as a second encoding, and reserve the green ramp for single-series/sequential charts. The sunburst has the same problem for sibling slices.
4. **The palette is hardcoded in ~25 places** across `callbacks.py` (and again in `applyTheme`, `styles.css`, and `theme_clientside.js`). Since `applyTheme` already applies the theme globally on every render, all the per-callback `is_dark` branching is **redundant** — deleting it removes ~150 lines and fixes dark-mode inconsistency at the same time. Leave only `type`, `categories`, `series`, and `tooltip.pointFormat` server-side.
5. **The daily chart silently hides data.** `render_daily_heatmap` drops every day where the top song had only 1 play (`callbacks.py:~1545`). It is a 365-point line pretending to be a heatmap. The most improvable visual — see "New visuals" #1. Also remove the dead grid-fallback branch (`~1560-1620`).
6. **The Social tab polls every 0.6 s forever** (`dcc.Interval(interval=600)`, `layouts.py:379`) to read a global set by the Venn click handler, and keeps polling while you are on other tabs. Replace with `dash_clientside.set_props` and delete the Interval.
7. **Duplicate titles.** An `H3` sits above a chart whose own `title` is the same string (`layouts.py:204` vs `callbacks.py:~1610`; repeated for Genre Trends at `~2090`). Set chart `title: None`.
8. **Unexplained metrics.** "Percentage" (of what?), "Top Song Play Count", "Unique Tracks", "Average Hours per Day", and "Exclude Parent Genres (Level 0)" have no tooltips. Given #6 and #11, the percentage label is not merely unclear but wrong.

**Also worth doing:** no URL/shareable state (`dcc.Location` + query params for tab/user/year — today everything is `persistence_type="local"`, so the app opens in a state from weeks ago); inconsistent time models across tabs (Wrapped is a year dropdown, Trends and Social have separate ranges); the date-preset state machine is duplicated verbatim for Trends and Social (`callbacks.py:409-486` vs `488-555`) and "Last 2y" uses `365*2` days, wrong across a leap year; `.2g` formatting renders large percentages as `1.2e+01`; `DataTable` is unthemed in dark mode and has `row_selectable`/`column_selectable` wired to nothing; Trends tables sit outside any `dcc.Loading`; the theme switch's `onLabel="☀️"` shows when **dark** is active (backwards) and `handle_theme` has a circular store↔toggle dependency; the stats table (`callbacks.py:1420`) duplicates `stats.py::create_stats_table`; mobile has exactly one media query (`styles.css:138`) while `marginLeft` runs to 210 px and chart heights are fixed.

**Dead front-end code:** `assets/theme_clientside.js:1-130` is Plotly restyle code — **[Verified]** nothing references `dash_clientside.theme`. Only its `highcharts` stub and `social.get_selected_region` (~140-170) are live. `assets/heatmap.js` is a hand-written stub not even listed in `MODULE_URLS`.

**Accessibility:** the Highcharts `accessibility` module is not loaded, so no chart has keyboard navigation or a screen-reader data table. `html.Label(...)` is never associated with its control (no `htmlFor`), so dropdowns are unnamed to screen readers. The ☰ button and theme toggle are icon-only with no `aria-label`; tab labels lead with emoji. Color is the sole differentiator in multi-series lines. White on `#1DB954` is ≈2.4:1 and **fails WCAG AA** — use `#158a3e` for selected pills and buttons. Loading overlays have no `aria-live`.

---

## P3 — New visuals the data already supports

All **[New]**, ranked by delight-per-hour. Every one feeds from existing tables.

1. **Calendar heatmap** (GitHub-style) of minutes/day with a year selector — `fact_plays.ts` + `ms_played`. Directly replaces the lossy daily chart in UX #5.
2. **Hour-of-day × weekday heatmap** — `fact_plays.ts` alone. Add a "night owl %" chronotype line.
3. **New-artist discovery curve** — `MIN(ts)` per artist over `v_plays_enriched`: new artists/month plus cumulative uniques. Yields the headline stat "you discovered 212 artists in 2024." Artist longevity (first-to-last play span) falls out of the same query.
4. **Streaks and binge sessions** — longest consecutive-day streak; sessions by gap-splitting `ts` at >30 min; session-length distribution.
5. **Sparkline stat cards** replacing the 2-column HTML stats table — the daily series is already in the store. A skip rate works off `ms_played < 30s` as a proxy.
6. **Genre share as a stacked area / streamgraph** with `stacking: "percent"` on a **datetime** axis rather than month-string `categories` (which hides zero-play months and breaks zoom and tooltips). Fixes the geometry problem *and* makes #6's percentages actually sum to 100. Add `zoomType: "x"` or a navigator, and cap the shared tooltip at top-N.
7. **Bump chart / rank race** of monthly top-5 artists — `get_artist_trends` already returns the data.
8. **Era histogram** — `dim_albums` release year vs play count ("you listen to 2016 music most"), plus average song-age over time. Explicit-% and duration distributions come from `dim_tracks`.
9. **Shannon entropy** of the artist/genre distribution as a monthly diversity line (see #14 — there is currently no diversity measure at all).
10. **Year-over-year compare mode** on Wrapped (year A vs B with ▲▼ rank deltas), **Jaccard taste-overlap matrix** for Social, PNG export via the Highcharts exporting module, and **cross-filtering** (click a bar or slice to set the global artist/genre filter).

---

## P3 — Code structure

### Duplication

- **The connection/registration block is repeated 10 times** — **[Jun #4]**. The `close_conn = False` / lazy `import duckdb` / `try…finally con.close()` / `unregister`+`register` pattern at `metrics.py:42, 259, 567, 683, 903, 1051` and `trends.py:49, 136, 277, 421`. June's suggested shape is right:

  ```python
  @contextmanager
  def duck_connection(con=None, db_path=None): ...
  def register_temp(con, name, df): ...
  ```

  This is the single biggest source of line count in the metrics package and removes several hundred lines.
- **Duplicated logic across modules** — **[Jun #5]**. `_register_list_param` (`preprocessing.py:188`, `social.py:28`); `_to_tuple` (`application.py:105`, `preprocessing.py:431`); track-ID extraction with *different* fallbacks (`db_ingest.py:44` vs `metrics/utils.py:4`, and a third partial copy at `api/utils.py:85`); and the `artist_genres_agg` "Child (Parent1, Parent2)" CTE **verbatim in three places** (`application.py:63`, `preprocessing.py:369`, `metrics.py:957`) — make it a database view, since it joins only DB tables. Plus the three base-filter implementations now drifted (#7).
- **[New]** The "top 2 artists" `STRING_AGG(… ' plays)')` snippet appears 4× in `metrics.py` (`:639, 739, 780, 837`) and once in `trends.py:206`. The "load play-counts, derive track_id" block is copy-pasted 3× (`metrics.py:269, 577, 693`). Cache-dir resolution is repeated 3× in `api.py`. The three `fetch_*` methods are near-identical ~40-line blocks → one `_fetch_cached(category, ids, batch_size, api_fn)`.

### `callbacks.py` is 2,482 lines in one closure — **[New]**

Natural seams (everything currently shares `df` and helpers by closure): `common.py` (cache + key builders + a `db()` context manager + `_fmt_genres`/`_truncate_label`/`_compute_left_margin`/`_merge_selected`), `filters.py` (collapse, the 7 search callbacks, date presets), `wrapped.py` (`:557-1437`, currently split in two by the Social block), `social.py` (`:772-1051`), `trends.py` (`:1656-2328`), `theme.py` (`:2330-2482`). Move figure construction into `dashboard/figures/` as **pure functions returning option dicts**, which makes them unit-testable without Dash — currently nothing tests a figure builder.

Three copy-paste clusters are worth factoring rather than merely moving:

- The **7 dropdown-search callbacks** (`:169-407`) are identical but for a column and an ID → one `make_search_callback(id, col, entity)` factory, ~35 lines each → ~40 total.
- The **date-preset machines** ×2 (above) → `resolve_preset(preset, min_ts, max_ts)`.
- The **artist/track/genre trend callbacks** (`:1968-2328`) are near-identical ~120-line blocks → one `build_trend_figure(...)`. Likewise the 8–9 clientside `render_single` registrations (`:2380-2470`) differ only by ID — loop over a list.

Also: deferred function-level imports (`from dash import html` at `:860`, `:1426`) hint at past circular-import issues — hoist them. `suppress_callback_exceptions = True` plus a hand-built `validation_layout` (`application.py:112-175`) is fragile.

### Fragile SQL assembly — **[Jun #6]**

`social.py:285` injects the date filter by string-replacing the first `"WHERE "` in generated SQL:

```python
.replace("WHERE ", f"WHERE 1=1{date_clause} AND ", 1)
```

This works only because `_base_plays_sql` happens to emit exactly one `WHERE` first. Any refactor silently breaks date filtering **and** misaligns the positional `?` params. Pass the clause into `_base_plays_sql` as a proper argument.

### Error swallowing — **[Jun #3]**

**49 bare `except Exception` blocks** across `src/` and `dashboard/` silently substitute defaults (empty tuple, empty DataFrame, skipped fallback). Cumulative effect: data bugs present as "the chart is empty" or "genres are missing" with nothing in the logs. At minimum each should `logging.warning(..., exc_info=True)` before falling back; most should narrow to the exception they actually defend against. Examples: `metrics.py:122, 867, 984, 1004, 1019`; `callbacks.py:636, 1308, ~2020`; `social.py:253`; and `api.py:1166, 1292, 1449` plus `api/utils.py:59`, which skip corrupt cache files without counting them.

**[New]** and compounding: `logging.basicConfig` is never called, so even the handlers that log are invisible. Fix that first — it makes the rest diagnosable.

### Unbound variable in `create_app` — **[Jun #8]**

`application.py:150-159`: if `get_db_connection()` raises, the `finally` calls `conn.close()` on a never-bound name — a `NameError` hidden by `contextlib.suppress(Exception)`. Initialize `conn = None` before the `try` and guard with `if conn is not None`. The same shape recurs in several callbacks (see Performance).

### Dead code — **[Jun #7 + New]**

`load_api_data`, `SpotifyData`, `_list_to_dict`, `get_rate_limit_status`, and the `__main__` block, which duplicates `load_api_data` almost line-for-line (`api.py:460, 247, 1536`) — **but see P0 #2 before deleting**; `_upsert_track`, `_insert_play`, `SPOTIFY_TRACK_NAMESPACE` (`db_ingest.py:103, 135, 41`, ~95 lines of superseded row-at-a-time code); `_circuit_breaker_triggered_at` (written, never read); `populate_missing_track_isrcs` + `populate_duration_and_explicit` and their `ingest-isrcs` / `ingest-duration-explicit` commands (superseded by `populate_track_metadata`).

**[New]** additions: `add_api_data` (`preprocessing.py:33`) has **no callers anywhere** — delete it, which also retires **[Jun #17]** (it mutates its input in place while returning it); `src/api/utils.py` is a second click CLI, unregistered in `pyproject.toml`, hardcoding `data/api/cache` and ignoring `SPOTIFY_API_CACHE_DIR`; `src/wrapped.py` is a legacy Excel export with a hardcoded folder and 2023-2024 range (the only consumer of `filter_songs`, which June #7 asked to mark deprecated); `src/link_artist_genres.py` is a duplicate entry point for the CLI command; and the non-DuckDB `else` branches at `metrics.py:256, 564, 678` are unreachable migration leftovers.

### Minor nits — **[Jun #22]**

`LIMIT {int(limit)}` / `rn <= {int(...)}` interpolation (`metrics.py:64, 638, 924`) is safe via the `int()` cast but should use DuckDB params for consistency; `_chunk_list` (`api.py:148`) is an instance method where `itertools.batched` or a module function would do; `tests/__init__.py` is unnecessary with pytest rootdir config; `trends.py:86` does a `strptime(month||'-01')` round-trip.

---

## P3 — Tests

**[Jun #20, quantified]** — **17% total coverage** (measured with `pytest-cov`), 7 passing tests for 11,083 lines:

```
src/cli.py                 0%      src/db_ingest.py           0%
src/io.py                  0%      src/load_genre_taxonomy.py 0%
src/wrapped.py             0%      src/link_artist_genres.py  0%
src/api/api.py             9%      dashboard/callbacks        3%
src/preprocessing.py       7%      src/metrics/trends.py     21%
```

**The entire ingestion pipeline — which produces the data everything else depends on — has zero tests.** That is precisely why P0 #1 and #2 went unnoticed for four months. Existing tests assert only column names and `>= 0` bounds, never values; nothing covers `get_filtered_plays`, `get_genre_trends`, `get_artist_trends`, `get_track_trends`, `get_playcount_by_day`, `get_genre_sunburst_rows`, `extract_track_id`, or `_deduplicate_albums_fuzzy`.

As June noted, the ingest functions are well-factored for testing (DataFrame-in → DB-out with a `db_path` argument); a tmp-path DuckDB fixture makes these cheap. Highest value first:

1. **`init-db` via Click's `CliRunner` on a tmp path, run twice** — would have caught P0 #1 outright.
2. **`load_history_into_fact_plays`** with a tmp DuckDB: counts, second run inserts 0 (idempotency), episode rows skipped, bad `ts` and NaN URI handled.
3. **`_to_timestamp` / `_extract_spotify_track_id` units**: `NaT`, NaN, None, int, tz-aware, `open.spotify.com?si=` URLs, `spotify:episode:` (currently returns the ID via the colon fallback).
4. **Cache behavior, no network**: mock the client to return `[None, {...}]` and assert the good item is still saved (P0 #3); write a truncated JSON file and assert recovery (P0 #4); assert `_save_cache` is atomic.
5. **`get_filtered_plays` boundaries**: a 23:00 play on the end day (#5), the skipped/incognito/unknown-reason filters, case-insensitive exclusion, and a multi-primary-artist track producing exactly one row.
6. **The MQPC/album-score math and `_deduplicate_albums_fuzzy`** extracted as pure functions — assert "Love" and "Love Songs" do **not** merge, and that "X (Deluxe)" and "X" do.
7. **`populate_*` against a pre-populated tmp cache** (`cache_dir=tmp_path`) with tiny fixture JSONs — DB rows and idempotency, no API calls.
8. **`_fetch_with_retry`** with a fake raising 429-then-success, `time.sleep` monkeypatched — assert backoff counts and that the lock is not held while sleeping.
9. **Rank/percentage in the trends functions**: ties, and that genre percentages sum to 100 (they do not — #6).

Also **[Jun #20]**: `conftest.py:8` runs Chrome/webdriver setup as an **autouse session fixture for every test run**, including pure-pandas unit tests. Scope it to the E2E module. (Commit `690f11c` added an early `if os.getenv("CI"): return None` but kept it autouse and session-scoped.)

---

## P3 — Docs and tooling

- **`CLAUDE.md` is wrong in ways that will mislead future agents** — it is loaded into every agent session, so fix it first:
  - Claims file-locked thread-safe caching (`:120`, `:175`). No file locking exists — only an in-process `threading.Lock` at `api.py:96`. **[Jun #2]**
  - Documents the mood/rollup tables as architecture. Nothing writes them (#8). **[New]**
  - Says `ENVIRONMENT` gates `.env` overriding (`:166`). True for the dashboard and `conn.py`, but `api.py:20` and `api/utils.py:11` call `load_dotenv(override=True)` unconditionally. **[New]**
  - Omits `src/wrapped.py`, `src/io.py`, `src/link_artist_genres.py`, `llm/`, `scripts/bootstrap_worktree.sh`, and two CLI commands. **[New]**
- **README inaccuracies** — **[New]**. Documents 8 commands; the CLI registers **10** (`ingest-isrcs`, `ingest-duration-explicit`). `README.md:154` claims "All commands support `--limit` and `--cache-dir`" — **false**: `init-db` and `ingest-history` have neither, and `ingest-isrcs` has no `--cache-dir`. The env section omits `DATA_DIR`, which is required. There is **no install section at all** (no `uv sync`, no Python version) — that lives only in `CLAUDE.md`/`AGENTS.md`. Steps 2 and 5 do not work (P0 #1, #2).
- **`--user-id` help says "User UUID"** (`cli.py:108`, repeated in `AGENTS.md`) but real IDs are slugs (`ben.greenawald`). **[New]**
- **No screenshots.** For a dashboard project, 3–4 images at the top of the README would do more for comprehension than the entire CLI reference table — and `AGENTS.md:30` already requires a screenshot for UI PRs. **[New]**
- **No quickstart without data.** No sample or fixture dataset, and `SEED.sql` seeds users with zero plays, so a new user cannot see anything work before a 30-day Spotify export arrives. A tiny synthetic fixture would double as the test fixture the suite needs. **[New]**
- **`AGENTS.md` and `CLAUDE.md` overlap and disagree** (UUID vs slug, different command lists). Consolidate to one; make the other a pointer. **[New]**
- **`TASK.md` is a historical migration plan presented as current.** **[FIXED `aa8e5df`: moved to `docs/TASK.md` with a superseded header.]** It references `music_schema_proposed.sql`, `genres_taxonomy.yaml`, and `moods_taxonomy.yaml` — **none of which exist** — and weighs Streamlit vs Next.js options never taken. Archive under `docs/` with a "superseded" header, or delete. **[New]**
- **`.pre-commit-config-ci.yaml` is inert** — **[New, Verified]**. pre-commit.ci reads a `ci:` block *inside* `.pre-commit-config.yaml`; it does not read a separate file by that name. `autofix_prs` and `autoupdate_schedule` are currently no-ops. Move those keys into the real config and delete the file.
- **Version drift** — **[New, Verified]**. Pre-commit pins ruff `v0.12.10`; `uv.lock` resolves `0.12.12`. No Dependabot or Renovate. CI never runs pre-commit, so hooks and CI can drift independently.
- **Python version mismatch** — **[New, Verified]**. Local dev is **3.13.11**; the CI matrix is `3.10`–`3.12`. The interpreter you actually develop on is never tested, and there is no `.python-version`. The CI `push` trigger also still lists a stale `code-quality` branch.
- **No type checker and no coverage.** Nothing in `pyproject.toml`, CI, or the hooks runs mypy/pyright/ty, and `.gitignore` has coverage entries nothing produces. Ruff has no `D` rules, so the (good) Google-style docstring consistency is unenforced. **[New]**
- **No `.env.example`** despite two required secrets. Undocumented env vars in active use: `DATA_DIR`, `MUSIC_DB`, `SPOTIFY_API_CACHE_DIR`, `SPOTIFY_API_RATE_DELAY`, `ENVIRONMENT`, plus `LLM_PROVIDER`, `GEMINI_MODEL`, `OPENAI_MODEL`, `GOOGLE_API_KEY`, `OPENAI_API_KEY` for `llm/`. (The real `.env` is correctly gitignored and absent from git history.) **[New]**
- **`pyproject.toml` hygiene** — **[Jun #19 + New]**. `description = "Add your description here"` is still the placeholder; no license, authors, classifiers, or URLs. Dependency audit by grep: `pyinstaller` — **zero imports anywhere**, remove; `webdriver-manager` — only `tests/conftest.py`, move to dev; `openai` + `google-genai` — only `llm/`, move to an optional extra; `openpyxl` — never imported but needed implicitly by `pandas.to_excel` in `src/wrapped.py:36`, so keep it and declare why; `highcharts-core` (dev) — imported by no Python, remove. No *missing* direct deps (June wondered about `plotly`; it is unused — the charts are Highcharts). Also: both `pytest.ini` and a potential `[tool.pytest.ini_options]` — keep one.
- **Add a `slopify status` (or `doctor`) command** — **[New]**. Nothing today verifies an ingest worked: no row counts, no orphan `fact_plays.track_id` check, no count of tracks missing metadata or bridge rows. Combined with "zero-ingest exits 0," a new user cannot distinguish a successful run from a totally failed one. **This is the largest operability gap after P0.**
- **Add a `slopify pipeline` command** running the documented sequence with `--dry-run`, `--skip-api`, `--force-refresh`, `--workers`, and a closing status summary. The 8-step manual sequence is the main onboarding friction once P0 is fixed. `--limit` also means different things per command (tracks vs JSON files). **[New]**
- **Repo hygiene is clean** — **[Jun #21, FIXED]**. `data.zip`, `tmp.zip`, and `egg-info` are gone; `temp.ipynb` and `genre_taxonomy_results.json` exist but are gitignored and untracked. Nothing build, cache, zip, database, notebook, or `.env` is tracked, in the tree or in history. `git ls-files` and `git status` now work (June could not run them). Fixed by commit `a79e113` — **the only one of the 22 findings resolved.** Minor leftovers: `llm/all_subgenres.txt` (48 KB) is tracked generated data that probably belongs in the gitignored `resources/`; the stale `slopify_wrapped.egg-info` directory is gitignored but still on disk; and `data/api/cache` at 7.9 GB / 517k inodes deserves a README note about disk cost.

---

## What's already good — don't "fix" these

- **DuckDB query performance.** 0.67 s for a full 525k-row scan. Resist optimizing SQL; the costs are elsewhere.
- **Per-callback connections.** 68 ms/interaction. Tidy for structure and the unbound-`con` bug, not for speed.
- **Transactions are used correctly** in the `ingest-history` path, and the bulk anti-join dedup is a good design. Re-ingest idempotency genuinely works as the README claims.
- **Resume mostly works by design** — each command selects NULL rows and the cache persists. Only dead-ID batches (P0 #3) and corrupt cache files (P0 #4) block it.
- Server-side dropdown search with `_merge_selected` keeps selections visible without shipping huge option lists — good pattern, just deduplicate the 7 copies.
- `SimpleLRUCache` keyed by filter signature is the right idea for a single-user app (it needs a lock and a better key hash, but the design is sound).
- `get_db_connection` defaulting to read-only with a clear error on a missing DB (`conn.py`).
- Per-chart `dcc.Loading` with `minHeight` on the Wrapped cards prevents layout jump.
- Local Highcharts assets (no CDN) with a defensive namespace stub preventing early clientside errors.
- Existing `:focus-visible` styling and dark-theme CSS variables (`styles.css:2-20`) — extend them to drive chart colors rather than replacing them.
- **Docstring discipline across `src/` is genuinely excellent** — keep it; just fix the handful that drifted **[Jun #10]**: `metrics.py:142` documents threshold 90 while the caller passes 70; `api.py:63` says 0.5 while the constant is 1.0; `application.py:125` documents a `DATA_DIR` error it no longer raises; `preprocessing.py:234` says `artist_genres` is "currently empty tuple" (it is populated); `cli.py:68` claims init-db is safe to re-run (P0 #1).
- Real, non-trivial features worth keeping: the sunburst genre hierarchy with "Child (Parent)" labels, Venn click-to-drill, expandable album track details, CSV export, and filter persistence (which needs the Reset button from UX #1 to be pleasant).

---

## Suggested order

June's suggested order started with the API null-batch fix and the connection-helper refactor. That is still reasonable, but two things now come first, and the refactor drops below correctness.

**1. Make it work (half a day).** P0 #1 (one line plus `IF NOT EXISTS`), P0 #2 (wire artist/album fetch into the CLI *before* deleting the dead code it hangs off), then P0 #3 and #4 — those two corrupt state permanently, so every further run compounds them.

**2. Make the numbers right (a day).** #5 (end-date, affects every chart), #6 (genre percentages), #7 (unify the three drifted filter builders). Then decide #8 — populate or delete the mood/rollup schema — and correct `CLAUDE.md` to match. #9 (`get_top_albums`) if you rely on the album view.

**3. Make failure visible (half a day).** Add `logging.basicConfig`, then `slopify status`. Do this before the error-swallowing cleanup (June #3) — narrowing 49 handlers is pointless while every log line is discarded.

**4. Lock it in (a day).** Tests 1–6 above, starting with `init-db` run twice and the `get_filtered_plays` boundary. This is the step that stops P0-class bugs recurring.

**5. Make it fast and pleasant.** Remove the 318 MB base DataFrame and drop `theme_data` from the server Inputs — two contained changes, the biggest perceived speedup. Then UX #1–#8, the colorblind palette, and the calendar and hour×weekday heatmaps.

**6. Housekeeping (an hour, any time).** Add a `LICENSE`; scrub the real names from `SEED.sql`; add `.env.example`; fold `.pre-commit-config-ci.yaml` into the real config; add 3.13 to the CI matrix; prune the misplaced dependencies.

**7. Then refactor.** Split `callbacks.py`, extract the connection context manager (June #4) and the shared SQL/utils (June #5), fix the `social.py` WHERE-injection (June #6), delete the dead-code list (June #7), and clear the remaining nits (June #22).

---

## Traceability: the 22 June findings

Verified individually against the current tree. **21 still present, 1 fixed.**

| Jun # | Finding | Status | Now in |
|---|---|---|---|
| 1 | Null item drops a whole 50-track batch | Still present | P0 #3 |
| 2 | Docs claim file-locked caching; code has none | Still present | P0 #4, Docs |
| 3 | ~50 silent `except Exception` blocks | Still present | Code structure → Error swallowing |
| 4 | Duplicated connection/registration boilerplate | Still present | Code structure → Duplication |
| 5 | Duplicated logic across modules | Still present | Code structure → Duplication; P1 #7 (now drifted) |
| 6 | Fragile `.replace("WHERE ")` SQL assembly | Still present | Code structure → Fragile SQL |
| 7 | Dead code and superseded commands | Still present | Code structure → Dead code (**read P0 #2 first**) |
| 8 | `conn` possibly unbound in `create_app` finally | Still present | Code structure → Unbound variable |
| 9 | Retry/backoff defects | Still present | P2 Ingest → Retry/backoff |
| 10 | Docstring/code mismatches | Still present | What's already good (docstrings); P1 #9 (the 70 vs 90 threshold) |
| 11 | Whole-table scans in `populate_track_artists` | Still present | P2 Ingest |
| 12 | Full table in memory; JSON to browser; LRU not thread-safe | Still present | P2 Performance (quantified: 318 MB / 21 MB) |
| 13 | Naive semicolon split in `_maybe_apply_ddl` | Still present | **P0 #1** — a live crash, not a latent hazard |
| 14 | Per-callback `duckdb.connect` churn | Still present | P2 Performance (**downgraded**: 68 ms/interaction) |
| 15 | `avg_hours_per_day` divides by full calendar month | Still present | P1 #11 |
| 16 | `io.py` robustness | Still present | P2 Ingest |
| 17 | `add_api_data` mutates its input | Still present | Code structure → Dead code (**no callers; delete**) |
| 18 | Second-resolution dedupe key undocumented | Still present | P2 Ingest |
| 19 | Dependency hygiene in `pyproject.toml` | Still present | Docs and tooling |
| 20 | Test coverage gaps; autouse session fixture | Still present | P3 Tests (quantified: 17%) |
| 21 | Repo hygiene (zips, egg-info, broken `git ls-files`) | **Fixed** (`a79e113`) | Docs and tooling (noted as fixed) |
| 22 | Minor nits | Still present | Code structure → Minor nits; P2 Performance (cache key); P1 #14 (Venn cap) |

Three June findings were **re-scoped** by October's measurements or greps rather than simply restated: **#13** proved to be a hard failure blocking all first-time setup (promoted to P0 #1); **#14** measured at 68 ms per interaction (demoted from a latency fix to tidiness); and **#17**'s `add_api_data` turned out to have no callers at all, so the mutation concern is moot — delete the function.
