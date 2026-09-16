# 双通道动态 PETC：N-sweep 实验包

本目录保存固定参数下的链长扫描实验。实验只改变智能体数量 `N`，没有针对不同链长重新调节控制器、LESO 或事件触发参数。

## 实验设置

- `N = [5, 10, 15, 20]`
- 方法：`Periodic`、`Dynamic-Dual`
- 随机种子：`0:19`
- 总运行数：`4 × 2 × 20 = 160`
- 仿真时长：`T = 40 s`
- 统计窗口：`[20, 40) s`
- 嵌套初值：每个 seed 先生成 `Nmax = 20` 的初值，再截取前 `N` 个智能体

冻结参数：

- `kp = 1`, `kd = 3`
- `omega_o = 20`
- `rho_c = rho_u = 0.9`
- `chi_c = chi_u = 0.5`
- `nu_c = nu_u = 1`
- `delta_c = delta_u = 2e-4`
- `eta_c(0) = eta_u(0) = 0.01`

## 核心结果（20 seeds 均值）

| N | Method | RMSE all | RMSE tail | Peak | Rc | Ru |
|---:|---|---:|---:|---:|---:|---:|
| 5 | Periodic | 0.043279 | 0.052013 | 0.074379 | 0 | 0 |
| 5 | Dynamic-Dual | 0.042291 | 0.050674 | 0.084362 | 75.63% | 95.56% |
| 10 | Periodic | 0.056135 | 0.079445 | 0.109940 | 0 | 0 |
| 10 | Dynamic-Dual | 0.055095 | 0.079158 | 0.117734 | 76.16% | 94.61% |
| 15 | Periodic | 0.072990 | 0.109563 | 0.151801 | 0 | 0 |
| 15 | Dynamic-Dual | 0.073306 | 0.112035 | 0.160779 | 75.90% | 94.73% |
| 20 | Periodic | 0.093773 | 0.159021 | 0.226951 | 0 | 0 |
| 20 | Dynamic-Dual | 0.095749 | 0.165306 | 0.240475 | 74.28% | 94.69% |

完整 mean ± std 见 `data/table_n_sweep_final.csv`。

## 目录说明

- `data/table_n_sweep_final.csv`：八组实验的 mean ± std 汇总
- `data/table_n_sweep_final.json`：参数、设置和结果的结构化记录
- `data/n_sweep_20seeds_raw.csv`：160 次运行的原始指标
- `data/n_sweep_edge_profiles_raw.csv`：逐 seed、逐边 RMSE
- `data/n_sweep_edge_profiles_summary.csv`：逐边 RMSE 的 mean ± std
- `figs/`：期刊图的 PNG 与矢量 PDF
- `matlab/run_n_sweep.m`：N-sweep 主程序
- `matlab/run_sim_metrics.m`：单次仿真和指标计算
- `matlab/petc_params.m`：冻结参数（正式默认规模仍为 `N=10`）

## MATLAB 复现

在本目录的上一级项目根目录运行：

```matlab
addpath('matlab');
run_n_sweep;
```

结果表明：随链长增加，全链和尾边误差平稳增大但保持有界；Dynamic-Dual 的通信节约率维持约 74%–76%，控制更新节约率维持约 94.6%–95.6%。
