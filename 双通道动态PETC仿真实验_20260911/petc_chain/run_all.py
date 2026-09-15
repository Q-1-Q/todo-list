"""Full first-version pipeline: gate -> A -> B -> C."""

from __future__ import annotations

from petc_chain import run_exp_a, run_exp_b, run_exp_c, run_gate


def main():
    rc = run_gate.main()
    if rc != 0:
        print("Stop: gate failed.")
        return rc
    run_exp_a.main()
    run_exp_b.main()
    run_exp_c.main()
    print("All experiments finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
