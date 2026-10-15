"""Rewrite the paper's prose with a plain-English style model, into a SEPARATE file. paper/main.tex is never modified.

The style model is Qwen2.5-7B-Instruct with the newsletter project's QLoRA adapter (adapter_v2). It was trained to
return {"items": [{"index", "headline", "body_html"}]} under a set of plain-English voice rules, so the prompt keeps
that format and those rules and replaces the sales task with "rewrite this paragraph".

Safety: every span that carries a fact (numbers and anything containing a digit, LaTeX commands, citations,
references, maths, code) is replaced by a placeholder like [[7]] before the model sees the text and restored
afterwards. A rewrite is rejected, and the original paragraph kept, if any placeholder is missing, duplicated or
invented, or if the length changes too much.

    python -m scripts.humanise_paper prepare               # laptop: paper/main.tex -> paper/humanise/units.jsonl
    python scripts/humanise_paper.py generate --adapter DIR  # GPU: units.jsonl -> outputs.jsonl
    python -m scripts.humanise_paper assemble              # laptop: -> paper/main_humanised.tex + report.md
"""
import argparse
import json
import re
from pathlib import Path

PAPER = Path("paper/main.tex")
WORK = Path("paper/humanise")
SKIP_ENVS = ("figure", "table", "lstlisting", "tabular", "tikzpicture", "equation", "align", "thebibliography")
MIN_WORDS = 20

# spans that must survive unchanged (order matters: commands with arguments first)
PROTECT = re.compile(
    r"\$[^$]+\$"                                                   # inline maths
    r"|\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})+"  # \cmd{...} with nested braces
    r"|\\[a-zA-Z]+\*?"                                             # bare command
    r"|\\[^a-zA-Z\s]"                                              # escapes: \% \& \, ...
    r"|(?<![\w])[\w+\-/×]*\d(?:[\w.,+\-/×]*[\w×])?"               # any word containing a digit (not its punctuation)
    r"|``|''"                                                      # LaTeX quotes
)
PH = re.compile(r"\[\[(\d+)\]\]")

SYSTEM = """You are an editor for a technical report written by an independent researcher. For each paragraph, rewrite it in plain, natural English with exactly the same meaning. Keep every fact. Do not add, remove or soften any claim. Tokens written like [[3]] stand for numbers, citations, model names, formulas or formatting: copy each one exactly once, unchanged, where it belongs in your sentence.

VOICE RULES (follow all of them; they are enforced by an automated check):

Register -- a researcher explaining their own work to a careful reader.
- Put the main point first. Never open with "It is worth noting", "What this means is", "It helps to".
- One hedge per sentence at most (usually / tends to / often / likely). Keep the hedges the paragraph already has.

Plain English -- the reader must take it in at first pass.
- Maximum 24 words per sentence. One idea per sentence.
- Maximum two clauses. Split at the "but", "which", "so that".
- No em dashes. Use a full stop or a comma.
- No sentence fragments. Every sentence needs a verb.
- Active voice.
- No more than two -tion/-ment/-ity nouns in a sentence. Prefer the verb.

Banned words and phrases: leverage, utilise, robust, seamless, pivotal, crucial, facilitate, unlock, empower, harness, ecosystem, landscape (figurative), game-changer, In conclusion, Ultimately.
Precise technical terms are fine and wanted. Precision is not jargon.

Never invent a number, date, name or result. Use only what the PARAGRAPH gives you.
No HTML tags. No em dashes."""


def protect(text):
    spans = []

    def sub(m):
        spans.append(m.group(0))
        return f"[[{len(spans)}]]"
    return PROTECT.sub(sub, text), spans


def restore(text, spans):
    return PH.sub(lambda m: spans[int(m.group(1)) - 1], text)


def units_from_tex(lines):
    """-> list of (start_line, end_line_exclusive, prefix, body): prose paragraphs and \\item texts worth rewriting."""
    units, depth_skip, i = [], 0, 0
    in_doc = False
    while i < len(lines):
        ln = lines[i]
        if r"\begin{document}" in ln:
            in_doc = True
        if r"\appendix" in ln:
            break
        for env in SKIP_ENVS:
            depth_skip += ln.count(r"\begin{" + env) - ln.count(r"\end{" + env)
        if not in_doc or depth_skip > 0 or not ln.strip():
            i += 1
            continue
        s = ln.strip()
        if s.startswith(r"\item "):
            prefix, body = ln[: ln.index(r"\item ") + 6], ln[ln.index(r"\item ") + 6:]
            units.append((i, i + 1, prefix, body))
            i += 1
            continue
        if s.startswith("\\") and not s.startswith(r"\paragraph") and not s.startswith(r"\textbf") and not s.startswith(r"\emph"):
            i += 1
            continue
        j = i                                              # a paragraph: consecutive plain lines
        while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith((r"\begin", r"\end", r"\item", r"\section", r"\subsection")):
            j += 1
        text = " ".join(l.strip() for l in lines[i:j])
        m = re.match(r"(\\paragraph\{[^}]*\}\s*)(.*)", text)
        prefix, body = (m.group(1), m.group(2)) if m else ("", text)
        units.append((i, max(j, i + 1), prefix, body))
        i = max(j, i + 1)
    return [u for u in units if len(u[3].split()) >= MIN_WORDS]


def prepare():
    lines = PAPER.read_text(encoding="utf-8").splitlines()
    # the abstract is inside \begin{abstract}..\end{abstract}: treat it as a paragraph too
    units = units_from_tex(lines)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "units.jsonl", "w", encoding="utf-8") as f:
        for k, (a, b, prefix, body) in enumerate(units):
            masked, spans = protect(body)
            f.write(json.dumps({"id": k, "start": a, "end": b, "prefix": prefix, "original": body,
                                "masked": masked, "spans": spans}, ensure_ascii=False) + "\n")
    print(f"{len(units)} prose units -> {WORK / 'units.jsonl'}")


def chatml(masked):
    user = (f"PARAGRAPHS:\n[0] {masked}\n\nReturn a JSON object {{\"items\":[ ... ]}} with one entry per paragraph index above. "
            "Each entry has keys: 'index' (integer), 'headline' (a short plain title), and 'body_html' (the rewritten "
            "paragraph, plain text). Keep every [[n]] token exactly once.")
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def generate(adapter, base, batch):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:                                              # older transformers spell it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter).eval()
    units = [json.loads(l) for l in open(WORK / "units.jsonl", encoding="utf-8")]
    out = open(WORK / "outputs.jsonl", "w", encoding="utf-8")
    for s in range(0, len(units), batch):
        chunk = units[s:s + batch]
        enc = tok([chatml(u["masked"]) for u in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=900, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for u, g in zip(chunk, gen):
            raw = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.write(json.dumps({"id": u["id"], "raw": raw}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{min(s + batch, len(units))}/{len(units)}", flush=True)


def parse_body(raw):
    try:
        j = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return j["items"][0]["body_html"]
    except Exception:
        return None


def validate(u, body):
    if body is None:
        return "output was not the expected JSON"
    body = re.sub(r"</?[bi]>", "", body).strip()
    want = list(range(1, len(u["spans"]) + 1))
    got = sorted(int(x) for x in PH.findall(body))
    if got != want:
        missing = sorted(set(want) - set(got))
        return f"placeholders changed (missing {missing[:6]}, found {len(got)} of {len(want)})"
    if "—" in body:
        return "contains an em dash"
    ratio = len(body.split()) / max(1, len(u["masked"].split()))
    if not 0.6 <= ratio <= 1.5:
        return f"length changed too much (x{ratio:.2f})"
    if any(ch in body for ch in "{}\\$"):
        return "introduced LaTeX characters"
    return None


def assemble():
    lines = PAPER.read_text(encoding="utf-8").splitlines()
    units = {u["id"]: u for u in map(json.loads, open(WORK / "units.jsonl", encoding="utf-8"))}
    outs = {o["id"]: o for o in map(json.loads, open(WORK / "outputs.jsonl", encoding="utf-8"))}
    new_lines, report, cursor, kept, changed = [], ["# Humanised rewrite: side by side\n"], 0, 0, 0
    for k in sorted(units):
        u = units[k]
        body = parse_body(outs.get(k, {}).get("raw", ""))
        why = validate(u, body)
        new_lines += lines[cursor:u["start"]]
        if why:
            new_lines += lines[u["start"]:u["end"]]
            kept += 1
            report.append(f"## Unit {k} (line {u['start'] + 1}): KEPT ORIGINAL, {why}\n\n{u['original']}\n")
        else:
            text = restore(re.sub(r"</?[bi]>", "", body).strip(), u["spans"])
            indent = re.match(r"\s*", lines[u["start"]]).group(0) if u["prefix"].startswith((" ", "\t")) or u["prefix"].lstrip().startswith(r"\item") else ""
            new_lines.append(u["prefix"] + text if u["prefix"] else text)
            changed += 1
            report.append(f"## Unit {k} (line {u['start'] + 1}): rewritten\n\n**Before:** {u['original']}\n\n**After:** {text}\n")
        cursor = u["end"]
    new_lines += lines[cursor:]
    Path("paper/main_humanised.tex").write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    report.insert(1, f"{changed} paragraphs rewritten, {kept} kept as the original (failed a safety check).\n")
    (WORK / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"rewritten {changed}, kept {kept} -> paper/main_humanised.tex, {WORK / 'report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "generate", "assemble"])
    ap.add_argument("--adapter", default="adapter_v2")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    {"prepare": prepare, "assemble": assemble}.get(a.step, lambda: generate(a.adapter, a.base, a.batch))()


if __name__ == "__main__":
    main()
