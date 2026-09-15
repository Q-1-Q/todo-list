# Dual-channel dynamic PETC on a leader-rooted chain

Python simulator for the self-consistent PETC theory draft. It does **not** reuse the old DoS / async ET-IESO experiment.

## Frozen choices

- Scheme A: `k=0` is initialized with fresh \(\bar\phi,\bar u\) and is not counted.
- Savings denominator: 1000 detection instants in the window `[10, 20)`.
- Five methods: Periodic, Static-Single, Static-Dual, Dynamic-Single, Dynamic-Dual (proposed).
- Communication variable is \(\varphi = k_p p + k_d v\).
- LESO uses the held applied input \(u[k]\).
- Leader node 0 sends \(\varphi_0\) for link 1.

## Run

```bash
cd /Users/nr/shuang
python3 -m pip install -r requirements.txt
python3 -m petc_chain.run_gate      # Lyapunov check + one trial
python3 -m petc_chain.run_exp_a     # proposed-method figures
python3 -m petc_chain.run_exp_b     # 5 methods x 10 seeds
python3 -m petc_chain.run_exp_c     # disturbance at nodes 1, 5, 10
python3 -m petc_chain.run_all       # all of the above
```

- Agent 1 applies sampled leader acceleration \(a_0(k)\) (ZOH). This cancels unmatched \(a_0\) at detection instants; intersample variation stays in \(r_h\).
- Initial states are the desired formation plus small random offsets. Absolute \(U(-1,1)\) positions make an \(N=10\) chain miss the \([10,20]\) s window.
