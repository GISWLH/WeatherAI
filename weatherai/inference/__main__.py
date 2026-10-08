"""python -m weatherai.inference cards [--json]          checkpoint cards / evidence levels
   python -m weatherai.inference check-fuxi-input DIR [--mask MASK]   validate official-layout FuXi-S2S inputs (no weights needed)"""
import argparse, json, sys

from . import cards


def main(argv=None):
    ap = argparse.ArgumentParser("weatherai.inference")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cards"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("check-fuxi-input"); p.add_argument("dir"); p.add_argument("--mask")
    a = ap.parse_args(argv)
    if a.cmd == "cards":
        if a.json:
            print(cards.to_json())
        else:
            rows = cards.summary_rows()
            keys = list(rows[0])
            print("\t".join(keys))
            for r in rows:
                print("\t".join(str(r[k]) for k in keys))
        return 0
    if a.cmd == "check-fuxi-input":
        from .fuxi_s2s import FuXiS2SInputs, check_inputs
        inp = FuXiS2SInputs.from_official_files(a.dir, a.mask)
        print(json.dumps(check_inputs(inp), indent=1))
        return 0


if __name__ == "__main__":
    sys.exit(main())
