#!/usr/bin/env python3
"""Before/after compilation PDF for a job dir.

BEFORE = job/originals/<file> (watermark present)  |  AFTER = job/finals/<file> (cleaned)
plus a zoomed watermark-band strip for both. Annotates each page with the Owner method and
the Audit decision pulled from state.json (the production veto gate), so the PDF is an honest
record of what was published vs held — not just a pretty diff.

Usage: python3 build_job_before_after.py <job_dir> [out.pdf]
"""
import os, sys, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

JOB = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "job_50"
OUT = sys.argv[2] if len(sys.argv) > 2 else JOB.rstrip("/") + "_before_after.pdf"
CELL, PAD = 520, 20


def imread(p):
    try:
        return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def to_pil(bgr, maxw):
    h, w = bgr.shape[:2]
    s = maxw / w
    return Image.fromarray(cv2.cvtColor(
        cv2.resize(bgr, (int(w * s), max(1, int(h * s)))), cv2.COLOR_BGR2RGB))


def band(bgr):
    H = bgr.shape[0]
    return bgr[int(0.40 * H):int(0.58 * H), :]


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def manifest_latest(path):
    latest = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    latest[r["id"]] = r
                except Exception:
                    pass
    return latest


# audit / owner status -> (badge text, color)
def badge(state, owner_status, audit_dec):
    if state == "PASS":
        if owner_status == "copied_no_watermark":
            return "PASS — no watermark (passed through)", (0, 120, 0)
        return "PUBLISHED — audit PASS", (0, 130, 0)
    if state == "REJECT":
        return f"HELD — audit {audit_dec or 'reject'}", (190, 0, 0)
    if owner_status == "auto_rejected":
        return "HELD — owner auto-reject", (190, 0, 0)
    return f"{state or owner_status or '?'}", (120, 80, 0)


def main():
    man = manifest_latest(os.path.join(JOB, "owner_manifest.jsonl"))
    st = {}
    sp = os.path.join(JOB, "state.json")
    if os.path.exists(sp):
        st = json.load(open(sp)).get("images", {})
    Fh, Fs, Fsm = font(16), font(15), font(13)
    pages = []

    origs = sorted(f for f in os.listdir(os.path.join(JOB, "originals"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    n_clean = n_passthru = n_held = 0

    for i, f in enumerate(origs, 1):
        iid = os.path.splitext(f)[0]
        rec = man.get(iid, {})
        sinfo = st.get(iid, {})
        state = sinfo.get("state")
        owner_status = rec.get("owner_status") or sinfo.get("owner_status")
        method = rec.get("method") or "-"
        audit_dec = sinfo.get("audit_decision")
        btxt, bcol = badge(state, owner_status, audit_dec)

        before = imread(os.path.join(JOB, "originals", f))
        if before is None:
            continue
        final_rel = rec.get("final")
        after = imread(os.path.join(JOB, final_rel)) if final_rel else None
        has_after = after is not None

        if state == "PASS":                      # authoritative: gate on the audit verdict
            if owner_status == "copied_no_watermark":
                n_passthru += 1
            else:
                n_clean += 1
        else:                                    # REJECT / held — never published
            n_held += 1

        pb = to_pil(before, CELL)
        bb = to_pil(band(before), CELL)
        if has_after:
            pa = to_pil(after, CELL)
            ba = to_pil(band(after), CELL)
        else:                                   # held: no published final — gray placeholder
            pa = Image.new("RGB", (CELL, pb.height), (238, 238, 238))
            ba = Image.new("RGB", (CELL, bb.height), (238, 238, 238))
            d0 = ImageDraw.Draw(pa)
            d0.text((16, pb.height // 2 - 10), "no published final\n(held for review)",
                    fill=(150, 60, 60), font=Fs)

        pw = PAD * 3 + CELL * 2
        ph = 10 + 24 + 22 + max(pb.height, pa.height) + 26 + max(bb.height, ba.height) + PAD
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        d.text((PAD, 8), f"[{i}/{len(origs)}]  {f[:74]}", fill="black", font=Fh)
        d.text((PAD, 8), " " * 0, fill="black", font=Fh)
        # status badge (right-aligned-ish on its own line below)
        y = 10 + 24
        d.text((PAD, y), "BEFORE  (original — watermark)", fill=(190, 0, 0), font=Fs)
        d.text((PAD * 2 + CELL, y), f"AFTER  ·  {method}", fill=bcol, font=Fs)
        d.text((PAD * 2 + CELL, y - 24), btxt, fill=bcol, font=Fsm)
        y += 22
        page.paste(pb, (PAD, y))
        page.paste(pa, (PAD * 2 + CELL, y))
        y += max(pb.height, pa.height) + 6
        d.text((PAD, y), "zoom — watermark band:", fill=(90, 90, 90), font=Fsm)
        y += 20
        page.paste(bb, (PAD, y))
        page.paste(ba, (PAD * 2 + CELL, y))
        pages.append(page)

    # cover summary page (first)
    import collections
    reasons = collections.Counter(
        (v.get("audit_decision") or "owner_auto_reject")
        for v in st.values() if v.get("state") == "REJECT")
    cover = Image.new("RGB", (PAD * 3 + CELL * 2, 440), "white")
    dc = ImageDraw.Draw(cover)
    dc.text((PAD, 24), "Watermark Removal — Before / After", fill="black", font=font(26))
    dc.text((PAD, 70), f"job: {os.path.basename(JOB)}   ·   {len(origs)} images", fill=(60, 60, 60), font=font(16))
    lines = [("published (cleaned, audit PASS):    " + str(n_clean), (0, 130, 0))]
    if n_passthru:
        lines.append(("no watermark, passed through:       " + str(n_passthru), (0, 120, 0)))
    lines.append(("held for review (not published):    " + str(n_held), (190, 0, 0)))
    for r, c in reasons.most_common():
        lines.append(("      • " + r + ":  " + str(c), (150, 60, 60)))
    lines += [
        ("", (0, 0, 0)),
        ("BEFORE = original catalog image (watermark present)", (30, 30, 30)),
        ("AFTER  = cleaned output, gated by an independent audit veto pass", (30, 30, 30)),
        ("Held items are conservative audit holds (protected text / unnatural", (30, 30, 30)),
        ("result), not silent failures — each page shows the verdict.", (30, 30, 30)),
    ]
    yy = 120
    for ln, col in lines:
        dc.text((PAD, yy), ln, fill=col, font=font(16))
        yy += 28
    pages.insert(0, cover)

    if not pages:
        print("NO PAGES")
        return
    pages[0].save(OUT, "PDF", save_all=True, append_images=pages[1:], resolution=100.0)
    print(f"PDF: {OUT} | pages: {len(pages)} | "
          f"cleaned={n_clean} passthru={n_passthru} held={n_held} | "
          f"size: {os.path.getsize(OUT)//1024} KB")


if __name__ == "__main__":
    main()
