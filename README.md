# Bulk Image Generator

Generates images in bulk from a YAML prompt list using **Nano Banana 2** (`gemini-3.1-flash-image`) via the Google AI Studio API.

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended)
- A [Google AI Studio](https://aistudio.google.com) API key

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies with uv:

```bash
uv pip install google-genai pyyaml
```

---

## Usage

```bash
# Use the default prompts.yaml in the same folder
python bulk_image_gen.py

# Point to a specific prompts file
python bulk_image_gen.py my_batch.yaml

# Use a separate prefix file (overrides prefix_file set inside the YAML)
python bulk_image_gen.py my_batch.yaml --prefix my_prefix.yaml
```

When run, the script will prompt you to paste your Google AI Studio API key (input is hidden). To skip the prompt, set it as an environment variable instead:

```bash
export GOOGLE_API_KEY="your-key-here"
python bulk_image_gen.py
```

---

## Prompts YAML format

The prompts file has three top-level keys:

| Key | Required | Description |
|---|---|---|
| `output_dir` | no | Folder to save images to. Defaults to `./output` |
| `prefix_file` | no | Path to a prefix YAML or .txt file (see below) |
| `prompts` | **yes** | List of prompt entries |

Each entry in `prompts` can be either a plain string or a mapping with these fields:

| Field | Required | Description |
|---|---|---|
| `prompt` | **yes** | The subject text sent to the model |
| `label` | no | Filename prefix. Auto-generated from the prompt if omitted |

### Minimal example

```yaml
output_dir: "output"

prompts:
  - prompt: "Subject: Broom of Flying"
  - prompt: "Subject: Decanter of Endless Water"
  - prompt: "Subject: Trusty Steed"
```

### With labels

```yaml
output_dir: "output"

prompts:
  - label: broom
    prompt: "Subject: Broom of Flying"

  - label: decanter
    prompt: "Subject: Decanter of Endless Water"
```

### With a prefix file

```yaml
output_dir: "output"
prefix_file: "prefix.yaml"

prompts:
  - prompt: "Subject: Broom of Flying"
  - prompt: "Subject: Decanter of Endless Water"
```

---

## Prefix file

A prefix is a block of descriptive text prepended to every prompt. Use it to set a consistent style, medium, lighting, or framing across your whole batch — like a "system prompt" for image generation.

The prefix and subject are combined like this before being sent to the model:

```
<prefix text>

Subject: Broom of Flying
```

The prefix file can be either a `.yaml` file with a `prefix:` key, or a plain `.txt` file.

### prefix.yaml example

```yaml
prefix: >
  Fantasy trading card illustration style. Painterly detail, rich jewel-toned
  colors, dark vignette border, dramatic studio lighting. The image depicts a
  single magical item floating against a deep black background. Render it with
  museum-quality realism — visible brushwork, aged patina, subtle arcane glow.
  No text, no labels, no people.
```

### prefix.txt example

```
Fantasy trading card illustration style. Painterly detail, rich jewel-toned
colors, dark vignette border, dramatic studio lighting. The image depicts a
single magical item floating against a deep black background. Render it with
museum-quality realism — visible brushwork, aged patina, subtle arcane glow.
No text, no labels, no people.
```

You can also pass the prefix file on the command line with `--prefix`, which overrides any `prefix_file` set inside the YAML:

```bash
python bulk_image_gen.py magic_items.yaml --prefix dark_fantasy.yaml
python bulk_image_gen.py magic_items.yaml --prefix watercolor.txt
```

---

## Output

Images are saved to the `output_dir` folder with filenames in this format:

```
001_broom_of_flying.png
002_decanter_of_endless_water.png
003_trusty_steed.png
```

The numeric prefix keeps them sorted in the order they appear in the YAML.

---

## File layout

```
bulk_image_gen.py   ← bulk image generation script
decompose_maps.py   ← D&D map transcription script
ranking.py          ← dungeon rating + HTML compile script
prompts.yaml        ← your prompt list
prefix.yaml         ← optional style/prefix block
output/             ← generated images saved here
MD-OPDC/            ← transcribed dungeon markdown by year
```

---

# One-Page Dungeon Ranker

Rates transcribed dungeon Markdown files via **DeepSeek V4 Flash** (thinking: high) on OpenRouter, appends a ranking block to the end of each `.md`, and can compile those blocks into a self-contained HTML summary.

## Usage

```bash
export OPENROUTER_API_KEY="your-key-here"

# Rate every .md in a year folder
python ranking.py MD-OPDC/2010

# Rate all year folders under MD-OPDC (2010, 2011, …)
python ranking.py MD-OPDC

# Remove trailing ranking blocks
python ranking.py MD-OPDC/2010 --clear-ratings

# Re-rate files that already have rankings
python ranking.py MD-OPDC/2010 --force

# Validate without calling the API
python ranking.py MD-OPDC/2010 --dry-run

# Compile rankings into HTML (table + card views)
python ranking.py MD-OPDC/2010 --compile
python ranking.py MD-OPDC --compile --output MD-OPDC/rankings.html
```

No extra dependencies — uses the Python standard library only.

## What it does

For each `.md` file in the target folder:

1. **Validates** that the file has a `## Transcription` section (i.e. it got past the thinking stage). Files stuck at `## Thinking` are skipped and listed in `errors.md` in that folder.
2. **Removes** the `## Thinking` section from files that have a transcription.
3. **Sends** the transcription to `deepseek/deepseek-v4-flash` via OpenRouter with `reasoning: high`, streaming thinking into per-file logs.
4. **Appends** a ranking block to the end of the file:

```
title: Prisoners of the Mountain King
summary: The player characters are captured by the kobold Mountain King and must escape his dungeon in an old dwarven mine without their starting equipment. An altar to the dwarven god Brimli grants a permanent +1 Constitution and 1500 XP to any character who speaks the god's name in Dwarvish.
rooms: 18
resolutions: Combat, Puzzles, Stealth, Roleplay, Traps, Exploration
concept_originality: 2
mechanics_originality: 3
interesting_details: 4
map_quality: 2
rated_at: 2026-06-28T06:01:34.536Z
model: deepseek/deepseek-v4-flash
```

`--compile` scans for trailing ranking blocks and writes a searchable HTML document with:

- Summary stats (count, per-category averages, overall average)
- Sortable table view with an **Average** column (mean of the four category scores)
- Card/list view
- Batch/year filter (works across `MD-OPDC/2010`, `MD-OPDC/2011`, etc.)

Each dungeon is scored out of **10** on four categories: **concept originality** (distance from a traditional dungeon crawl), **mechanics originality**, **interesting details**, and **map quality**. The model is instructed to be a strict but fair judge and use the full scale, including 1s and 2s.

| Flag | Description |
|---|---|
| `--compile` | Extract rankings and write HTML instead of calling the API |
| `--clear-ratings` | Remove trailing ranking blocks from `.md` files |
| `--output PATH` | HTML output path for `--compile` |
| `--log-dir PATH` | Per-file OpenRouter logs (default: `<folder>/logs`) |
| `--force` | Re-rate even when a ranking block already exists |
| `--workers N` | Parallel workers (default: 4) |
| `--no-parallel` | Force sequential rating |
| `--delay SECONDS` | Pause between files in sequential mode |
| `--dry-run` | Report what would be rated without API calls |

| Artifact | Location | Description |
|---|---|---|
| Ranking block | End of each `.md` | `key: value` lines appended after the transcript |
| `errors.md` | Target folder | Files skipped because they never reached `## Transcription` |
| `logs/ranking_<file>.log` | Target folder | Streamed thinking + content chunks from OpenRouter |
| `rankings.html` | Target folder (or `--output`) | Compiled summary from `--compile` |

---

# D&D Map Decomposer

Batch-transcribes one-page D&D dungeon maps (PDF, PNG, or JPEG) into Markdown using **Gemini 3.1 Flash-Lite** with medium reasoning and streamed thinking summaries.

## Usage

```bash
# Process every PDF/PNG/JPEG in a folder (writes to <maps_folder>/output/)
python decompose_maps.py /path/to/maps

# Write Markdown beside the source maps instead
python decompose_maps.py /path/to/maps --in-place

# Reprocess even when a matching .md already exists
python decompose_maps.py /path/to/maps --force

# Write Markdown elsewhere and keep logs in a custom folder
python decompose_maps.py /path/to/maps --output-dir ./transcripts --log-dir ./transcripts/logs

# Control parallelism
python decompose_maps.py /path/to/maps --workers 6
python decompose_maps.py /path/to/maps --no-parallel
```

Set `GOOGLE_API_KEY` to skip the interactive key prompt:

```bash
export GOOGLE_API_KEY="your-key-here"
python decompose_maps.py /path/to/maps
```

## What it does

For each map file, the script:

1. Sends the file plus this prompt to `gemini-3.1-flash-lite` at **medium** thinking level:
   > I have here a one page D&D dungeon concept. Start by describing the page, then transcribe all the written text, then describe the map with enough detail for someone who can't see it to play.
2. Streams the response, capturing both thinking summaries and final text.
3. Writes `<map_name>.md` to `<maps_folder>/output/` by default, or beside the source maps with `--in-place`.
4. If output is blocked (safety, token limit, prompt block, etc.), records the block reason and any partial thinking/text, then continues to the next map.

When the folder contains **more than 4 maps**, the script processes maps in parallel unless you pass `--no-parallel`.

### Parallelism

| Flag | Default | Description |
|---|---|---|
| `--workers N` | `4` | Maximum concurrent workers |
| `--no-parallel` | off | Force one-at-a-time processing |
| `--delay SECONDS` | `0.5` | Pause between maps in sequential mode only |

Worker count scales down automatically to match remaining work. For example, with `--workers 4` and 10 maps, the batch starts with 4 workers; as maps finish, later waves use 3, 2, or 1 worker as needed. With 5 maps and `--workers 4`, the last map runs on a single worker.

Folders with **4 or fewer maps** run sequentially by default. Use `--no-parallel` to force sequential mode even for large folders.

## Output

| Artifact | Default location | Description |
|---|---|---|
| Markdown transcript | `<maps_folder>/output/<map_name>.md` | Thinking (if streamed), transcription, and block notes |
| Per-map log | `logs/<map_name>.log` | Detailed send/stream/receive timeline for that file |
| Session log | `logs/decompose_session_<timestamp>.log` | Batch summary across all maps |

Use `--in-place` to write `.md` files next to the source maps, or `--output-dir` for a custom folder.

Existing `.md` files in the output location are skipped unless you pass `--force`.
