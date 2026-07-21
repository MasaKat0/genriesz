# weight screen pilot（2026-07-20 実施、2026-07-21 追試）

`REFERENCE_SELECTION_PLAN.md` §18.9 が引用する数値の一次資料。計画書の表は
`summary.txt`（本 pilot）と `followup_probes.txt`（追試 3 本）から取っている。

| ファイル | 内容 |
|---|---|
| `weight_screen_pilot.py` | 本 pilot（閾値 4 条件 × $t\in\{0,4\}$ × R=120、960 jobs、12 並列で約 27 分） |
| `analyse_pilot.py` | 集計。出力を固定したものが `summary.txt` |
| `probe_truth_ess.py` | 追試 1: 真の representer の ESS 比を design × overlap で実測 |
| `probe2.py` | 追試 2: oracle 劣化・regret>1 の fold 一覧・長さの分解 |
| `probe3.py` | 追試 3: hidden 項が $\alpha_0$ の ESS を動かすことの確認 |
| `followup_probes.txt` | 追試 3 本の出力を固定したもの |

## 再実行

リポジトリのルートから:

```bash
PYTHONPATH=src python3 notebooks/experiments/pilots/weight_screen_2026-07-20/weight_screen_pilot.py
PYTHONPATH=src python3 notebooks/experiments/pilots/weight_screen_2026-07-20/analyse_pilot.py
```

Parquet（約 19MB）は `notebooks/experiments/results/weight_screen_pilot/` に書かれ、
`.gitignore` 対象。seed は `Numerics.base_seed` から決定的に導出されるので再実行で
同一の数値が得られる。macOS/Windows では `__main__` ガードが必須
（`ProcessPoolExecutor`、`notebooks/experiments/README.md` 参照）。

固定した `.txt` は pandas 出力の行末空白を除去して保存している。再実行出力と
比較するときは `diff <(sed 's/[[:space:]]*$//' <再実行出力>) summary.txt` のように
行末空白を無視すること。

## 注意

- 結論は「screen 採用不能」（計画書 §18.9 追補）。スクリプトは棄却の記録として残す。
- 本 pilot（2026-07-20）は `effective_sample_ratio` の peak 正規化（2026-07-21）より
  前のコードで実行された。pilot の ESS 値はオーバーフロー領域に達していないため
  数値は変わらない。当時のコードで厳密に再現するには、本ディレクトリのスクリプトは
  残したまま `refsel` だけを当時の状態に戻す:

  ```bash
  git restore --source 65d80ab -- notebooks/experiments/refsel
  # …実行…
  git restore -- notebooks/experiments/refsel   # 元に戻す
  ```

  （コミット `65d80ab` 自体には本ディレクトリが存在しないため、checkout では再現できない。）
