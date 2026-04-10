# Bulk Image Generator

Generates images in bulk from a YAML prompt list using **Gemini 3.1 Flash** (Nano Banana 2) via the Google AI Studio API.

## Requirements

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com) API key

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
bulk_image_gen.py   ← the script
prompts.yaml        ← your prompt list
prefix.yaml         ← optional style/prefix block
output/             ← generated images saved here
```
