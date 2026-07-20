# Reference-based loss--link selection: 実験計画 v2

本書は `reference_selection_experiment_details.md`（v1）と `simulation_coding_design_v10.md`
を**置き換える**。対象は Main.tex §4（`sec:nested_selection_inference`）と §7.2
（`sec:selection_simulation_design`）の数値的裏付けである。

実装:

- `notebooks/experiments/refsel/`（パッケージ）
- `notebooks/experiments/09_reference_based_loss_link_selection.ipynb`
- `tests/test_reference_selection_experiment.py`

本書の数値はすべて**実測値**である。設計段階の見積りと食い違った箇所は §18 に記録した。

---

## 1. なぜ計画を作り直すか

v1 の実験は「提案手法を回して bias・RMSE・coverage を出す」構成だった。査読では通らない。
Econometrica の査読者が最初に投げる質問に対して、v1 は一つも答えを用意していない。

| 査読者の質問 | v1 の対応 |
|---|---|
| R1. なぜこの手法が要るのか。Bregman loss を CV すれば済むのでは | **対照実験なし**。§4 L794 の主張が未検証 |
| R2. Theorem 3 は*一様*被覆の主張だ。最悪ケースを見せてほしい | **固定 DGP 数点の平均被覆のみ**。least-favorable な系列がない |
| R3. reference allowance $b_r$ は肝心な部分を仮定で逃げている。外れたらどうなる | **misspecified reference の実験なし**。$c_r$ 感度のみ |
| R4. bias が実際に無視できるとき、bias-aware 区間は損をしないのか | 長さは記録するが**対比する設計がない** |
| R5. oracle inequality の remainder は情報量があるのか、それとも空虚か | regret は出すが**定理の remainder と突き合わせていない** |
| R6. 再現できるのか | **40 コア日・fast mode なし**。replication package として成立しない |

加えて v1 には実装上の問題があった（δ 予算超過、cross-fit bias-aware 区間に定理がない、
audit が outcome noise を含むため精度不足、integration audit が計算量の 8 割を占める）。
本計画はこれらも同時に解消する。

## 2. 設計原則

1. **一実験＝一主張。** どの定理・命題のどの部分を検証するかが言えない実験は載せない。
2. **一様性の主張には最悪ケースを報告する。** 平均被覆ではなく DGP 族上の**最小被覆**を主表に出す。
3. **競合手法を実装する。** 「提案手法は良い数字が出た」ではなく「素朴な代替は具体的にこう壊れる」を示す。
4. **候補の fit を全ルールで共有する。** 選択規則の比較は追加コストがほぼゼロ（§7.1）。
5. **段階的な計算量。** smoke / pilot / publication の 3 tier。査読者が 1 時間で回せる tier を必ず持つ。
6. **監査量は解析的に計算する。** simulation truth を使えるのだから outcome noise を混ぜない（§10）。

---

## 3. 検証する主張と実験の対応

| ID | 実験 | 検証対象 | 答える質問 | 実装 |
|---|---|---|---|---|
| E1a | generator 再スケール不変性 | §4 L794「raw Bregman は候補間で比較不能」 | R1 | `refsel.rescaling` |
| E1b | 選択規則の horse race | 同上（Monte Carlo 版） | R1 | `refsel.selection` / `report.selection_rule_table` |
| E2 | bias bound の妥当性と緊密性 | Thm `data_dependent_bias` の**両側** | R5 | `report.bias_bound_table` |
| E3 | oracle inequality の remainder | Thm `nested_oracle`, Cor `oracle_remainder` | R5 | `report.oracle_regret_table` |
| E4 | drifting bias 族上の一様被覆 | Thm `uniform_selected_inference` | **R2** | `report.uniform_coverage_table` |
| E5 | reference の頑健性と整合チェック | Prop `several_references`, eq `reference_check` | **R3** | `report.reference_robustness_table` / `reference_check_table` |
| E6 | bias が無視できる場合の区間長 | Cor `oa:bias_aware_length`, Cor `selected_wald` | R4 | `report.interval_length_table` |
| E7 | 高次元設計 | 規則が $d=50$ でも動くこと | 補助 | grid C |

E4 と E5 が主表になる。E1b は §4 の存在意義そのものなので主表に次ぐ扱い。

---

## 4. E1a: generator の再スケール不変性（決定的デモ）

Main.tex L794 の主張は次のとおり：

> rescaling a generator changes its objective value without changing the unregularized representer.

これは Monte Carlo を要しない**決定的な事実**であり、一つの表で片が付く。

`GRRGLM` が最小化するのは（`src/genriesz/glm.py:296-306`）

$$L(\beta)=\frac1n\sum_i\left[g^\*(X_i,\phi(X_i)^\top\beta)-M_i^\top\beta\right]+P_\lambda(\beta).$$

$g\mapsto\kappa g$ とすると $(\kappa g)^\*(v)=\kappa g^\*(v/\kappa)$ なので、$\lambda=0$ のとき

- 係数は $\beta^\*_\kappa=\kappa\beta^\*_1$ にスケールする、
- **fitted representer $\widehat\alpha$ は不変**（$(\kappa g)'^{-1}(\kappa v)=g'^{-1}(v)$）、
- 目的値および held-out Bregman 基準はちょうど $\kappa$ 倍になる。

**penalty の扱いが要点である。** $\beta=\kappa b$ を代入すると
$F_\kappa(\kappa b)=\kappa\left[\text{（無正則化基準）}(b)\right]+P_\lambda(\kappa b)$ なので、
**penalty が 1 次同次であれば不変性はそのまま生き残る**。本実験が使う $\ell_1$ penalty は
1 次同次である。すなわち incomparability は「正則化を落としたから」生じる人工物ではなく、
実際に推定している正則化付き推定量についてそのまま成立する。

**実測**（$\kappa\in\{0.5,1,2\}$、$n=2000$、second-order、L-BFGS `tol=1e-10`、
$\max|\Delta\widehat\alpha|$ は $\kappa=1$ 基準）：

| generator | penalty | $\max|\Delta\widehat\alpha|$（$\kappa=0.5/2$） | 目的値の比（$\kappa=0.5/2$） | held-out 基準の比 |
|---|---|---|---|---|
| SQ | $\ell_1$, $c=0$ | 0.00019 / 0.00016 | 0.500000 / 2.000000 | 0.499999 / 1.999999 |
| SQ | $\ell_1$, $c=1$ | 0.00102 / 0.00067 | 0.499999 / 2.000002 | 0.500001 / 1.999996 |
| SQ | $\ell_2$, $c=1$ | **0.556 / 0.883** | **0.515 / 1.895** | 0.503 / 1.968 |
| UKL | $\ell_1$, $c=0$ | 0.00052 / 0.00056 | 0.500000 / 2.000000 | 0.500001 / 2.000006 |
| UKL | $\ell_1$, $c=1$ | 0.0219 / 0.0079 | 0.500008 / 1.999983 | 0.500020 / 1.999975 |
| UKL | $\ell_2$, $c=1$ | **1.331 / 2.154** | **0.490 / 2.068** | 0.501 / 2.008 |
| BP(0.5) | $\ell_1$, $c=0$ | 0.00012 / 0.00026 | 0.500000 / 2.000000 | 0.499998 / 2.000008 |
| BP(0.5) | $\ell_1$, $c=1$ | 0.00197 / 0.00099 | 0.499826 / 2.000346 | 0.499993 / 1.999925 |
| BP(0.5) | $\ell_2$, $c=1$ | **0.949 / 1.329** | **0.679 / 0.893** | 0.522 / 1.788 |

$\ell_1$ の行では**同一の推定量**の held-out Bregman 基準を任意に大小できる。
$\ell_2$ の行では推定量自体が動くが、それは generator の再スケールが実効 penalty を
黙って変えたからであり、順位づけの根拠としてはやはり成立しない。
どちらに読んでも generator をまたぐ raw Bregman CV は統計的に無意味である。

**実装**: `refsel.candidates.ScaledGenerator`（`BregmanGenerator` を継承し
`g`/`grad`/`grad2`/`inv_grad`/`domain_binding` の 5 つを override）。
`SquaredGenerator` を継承してはならない — `glm.py:252` の閉形式分岐に落ちて誤った結果になる。
本性質は unit test で恒久的に固定してある。

---

## 5. E1b: 選択規則の horse race

同一 fold・同一の 90 候補 fit に対して、以下の規則を**すべて**適用する。fit を共有するので
追加コストはほぼゼロである（§7.1）。

| ID | 規則 | 何を分離するか |
|---|---|---|
| `proposed` | $\arg\min\ \widehat U_a^2+\widehat V_a^+/n_{\mathrm{eval}}$ | 提案手法 |
| `proposed_min` | 同上、ただし $\widehat U_a^{\mathcal R}=\min_r(\cdot)$ | Prop `several_references` |
| `bregman_cv` | diagnostic 標本上の**各候補自身の** Bregman 基準を最小化 | §4 が示す通り恣意的。素朴な代替 |
| `lsif_cv` | generator 非依存の squared risk $\tfrac12E[\alpha^2]-E[m(\alpha)]$ を最小化 | **強い対抗馬**。候補間で比較可能だが $\alpha$ の推定誤差を測るのであって目標母数の drift を測らない |
| `abs_drift` | $\arg\min|\widehat D_a|$（$q_a,b_r$ なし） | 同時信頼半径 $q_a$ の寄与を分離 |
| `score_var` | $\arg\min\widehat V_a^+$ のみ | bias を無視した場合 |
| `fixed_sq` / `fixed_ukl` / `fixed_bkl` / `fixed_bp05` | rich dictionary・$c=1$ 固定 | 実務家の既定選択 |
| `oracle` | $\arg\min$ audit risk（実行不能） | 下限 |

`lsif_cv` を入れるのが重要である。「generator ごとの目的値が比較不能なら、共通の squared risk で
比較すればよい」という自然な反論に対し、**それでも目標母数の bias は測れない**ことを数値で示す。

各規則について、選ばれた候補の RMSE・|bias|・oracle regret・区間の被覆と長さを報告する。
**被覆は無条件**である（§14）。`fixed_bkl` は ATE では常に利用不能になる（§18.2）。

---

## 6. DGP

### 6.1. Low-dimensional base design

v1 から変更なし。$Z\sim N(0,I_5)$、

$$h_L(Z)=0.6Z_1-0.4Z_2+0.5(Z_3^2-1)+0.3\sin(Z_4),\qquad e_0(Z)=\Lambda(sh_L(Z)),$$
$$\mu_0(Z)=1+Z_1+0.5(Z_2^2-1)+0.5\sin(Z_3)+0.25Z_4Z_5,\qquad \tau(Z)=1+0.5Z_1-0.25Z_2,$$
$$Y=\mu_0(Z)+D\tau(Z)+\varepsilon,\quad\varepsilon\sim N(0,1),\qquad \theta_0=1.$$

$s\in\{0.5,1.5,2.5\}$。

### 6.2. Drifting misspecification 族（E4 の中核）

一様被覆を検証するには、bias と standard error の比
$t=\sqrt{n_{\mathrm{eval}}}|B|/\sqrt V$ を**制御して掃く**必要がある。
bounded-normal-mean 問題が指標づけられているのはこの $t$ だからである。

$$\psi(Z)=\cos(2\pi Z_1),\qquad
h_L^{(b)}(Z)=h_L(Z)+b\,\psi(Z),\qquad \mu_0^{(b)}(Z)=\mu_0(Z)+b\,\psi(Z).$$

$\tau$ は触らないので **$\theta_0=1$ は $b$ によらない**（unit test で固定）。

$\psi$ の選択理由：$Z_1\sim N(0,1)$ 上で $\cos(2\pi Z_1)$ は多項式・$\sin$・$|\cdot|$ の
張る空間とほぼ直交する（無条件の $R^2=0.0219$）。

**ただし候補基底は treatment-specific であり、$D$ 自体が $\psi$ を含む propensity に依存するため、
交互作用が $\psi$ を部分的に吸収する。** 実測（$N=3\times10^5$）：

| $b$（対応する $t$） | 無条件 base | candidate rich（treatment-specific） | `LowOutcomeBasis` |
|---|---|---|---|
| 0 | 0.0219 | 0.027 | 0.000 |
| 0.406（$t=1$） | 0.0219 | 0.057 | 0.028 |
| 0.857（$t=2$） | 0.0219 | 0.146 | 0.113 |
| 1.383（$t=4$） | 0.0219 | **0.285** | 0.245 |

したがって「分散の 98% が表現不能」と言えるのは無条件射影の話であり、実際に効く空間では
最大でも 71% である。**これは実験を無効にしない**：$t$ は仮定ではなく §6.3 で
実測して校正しているので、部分的な吸収は校正値に既に織り込まれている。

守るべき不変条件は直交性ではなく**非対称性**である。同じ $b=1.383$ での実測：

| 空間 | $R^2$ |
|---|---|
| candidate rich | 0.285 |
| `correct` reference の outcome basis | **1.000000** |
| `correct` reference の propensity features | **1.000000** |
| `misspecified` reference の outcome basis | 0.251 |
| `misspecified` reference の propensity features | 0.000 |

unit test はこの非対称性を（無条件射影ではなく）実際に使う空間で固定している。

#### 6.2.1. 誰が $\psi$ を表現できるべきか（重要な設計上の分岐）

$\psi$ を「全員が表現できない」方向にすると**実験が壊れる**。実測（§18.1）では、
その設計だと reference 自身の drift が $B_r=0.649$ に達し、自らの allowance $b_r=0.111$ を
大きく破る。Thm `data_dependent_bias` の前提が全候補について同時に崩れるので、
$t>0$ の領域で測っているものが無意味になる。

正しい設計は次の非対称性である。

| 主体 | $\psi$ を表現できるか | 理由 |
|---|---|---|
| candidate dictionary（linear/second-order/rich） | **できない** | これが候補 bias $B_a$ を生む |
| candidate 共有 outcome estimator（`LowOutcomeBasis`） | **できない** | 同上。$B_a$ は積なので両方が誤っている必要がある |
| `correct` reference（propensity・outcome とも） | **できる** | 手法の前提。reference は「bound が既知の、より良い推定量」である |
| `misspecified` reference | できない | E5 が測る失敗ケース |

この非対称性のため、**reference は fold 共有の outcome estimator を使わず自前の outcome
series を持つ**（v1 は共有していた）。Main.tex L1125 の「correctly specified outcome series」
という記述とも整合する。

### 6.3. $b$ の校正

**benchmark specification を SQ・rich・$c=0$ に固定**し、
$t(b)=\sqrt{n_{\mathrm{eval}}}|B_{\mathrm{bench}}(b)|/\sqrt{V_{\mathrm{bench}}(b)}$
が目標値に一致する $b$ を求める。benchmark を固定するのは $b$ の定義が選択規則に
依存しないようにするためで、実際に選ばれた候補の $t$ はこれと異なる。
**その差こそが E4 の測定対象である。**

実測した校正曲線（low design, $s=1.5$, $R_{\mathrm{cal}}=40$, integration $10^5$）:

| $b$ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | 1.3 | 1.6 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| $t$（$n=1000$） | 0.05 | 0.13 | 0.41 | 0.85 | 1.60 | 2.47 | 3.59 | 5.08 | 6.77 |
| $t$（$n=3000$） | 0.01 | 0.25 | 0.97 | 2.12 | 3.49 | 5.29 | 7.77 | 11.16 | 14.82 |

単調で、目標 $t\in\{0.5,1,2,4\}$ はいずれもグリッド内に収まる。逆補間した $b$ は
`refsel/calibration.json` に**コミットする**。publication run は校正を再実行せずこの表を読むので、
本実行は決定的である。

### 6.4. High-dimensional design

v1 から変更なし（$d=50$、$\operatorname{Cov}(Z_j,Z_\ell)=0.5^{|j-\ell|}$、$s\in\{0.75,2.0\}$）。
drifting 族は同じ $\psi$ を加える（$t\in\{0,1\}$ のみ）。

---

## 7. Candidate library

| Component | Values |
|---|---|
| Generator | SQ (`C=0`)，UKL (`C=1`)，BKL (`C=1`)，BP(0.25)，BP(0.5)（いずれも ATE branch） |
| Dictionary | linear，second-order，rich |
| Penalty multiplier | 0，0.25，0.5，1，2，4（$\lambda=c\sqrt{\log\max(p,2)/n_{\mathrm{tr}}}$、$\ell_1$） |

計 90。BP(1) は fixed branch 上で $g''\equiv2$ となり SQ と同じ Bregman 幾何になるため除外する。
**実測で厳密に一致することを unit test で固定した**（v1 は「確認する」と書いて未実装だった）。

### 7.1. 計算構造（規則を増やしても fit は増えない）

1 つの fold で $(\text{dictionary kind},X_{\mathrm{tr}})$ が同じなら `ExperimentBasis` は
完全に同一であり、$(\text{loss},\omega)$ が同じなら generator も同一である。したがって：

- basis は fold あたり **3 個**（v1 は 90 回 fit していた）、
- generator インスタンスは **5 個**（v1 は 90 個。branch 符号は $X$ のみの関数なので共有すると
  キャッシュが効く）、
- integration 標本の生の特徴量を (design, overlap, $b$, dictionary) 単位でキャッシュし、
  fold ごとの標準化はアフィン変換のみ、
- 候補の $\widehat\alpha$ は $\Phi_{\mathrm{int}}B$（$B$ は $p\times30$）の 1 回の行列積、
- **選択規則・reference・allowance scale はすべてこの共有結果の上で走る。**

この 2 点（basis 共有と generator 共有＋branch cache）は unit test で固定してある。

---

## 8. Sample splitting

v1 と同一。5 folds を回転させ、fold $k$ を evaluation、fold $(k+1)\bmod5$ を diagnostic、
残る 3 folds を training とする。candidate fitting・outcome estimation・reference fitting は
training のみ、選択は diagnostic のみ、evaluation は選択後の score 評価のみに使う。

---

## 9. Reference と allowance

| ID | representer | outcome | $b_r$ |
|---|---|---|---|
| `truth` | simulation truth | simulation truth | 0 |
| `correct` | logistic（true index ＋ **$\psi$**） | 自前 series（`LowOutcomeBasis` ＋ **$\psi$**） | sandwich 楕円体の積 |
| `misspecified` | logistic（**$Z_1,Z_2$ の線形項のみ**） | 自前 series（**切片＋生 $Z$ のみ**） | correct と同じ式（＝過小） |
| `rff` | 2,000 random Fourier features の squared Riesz | fold 共有の gradient boosting | $c_r/\sqrt{n_{\mathrm{eval}}}$ |

`misspecified` は**allowance の式を正しいまま reference だけを壊す**。
allowance は pseudo-true 係数まわりの標本誤差を測るのであって近似誤差を測らないので、
これは意図的に過小になる。

### 9.1. Allowance のスケーリング

honest な $b_r$ に $\rho\in\{0,0.5,1,2\}$ を掛けたものも同時に評価する。$\rho=0$ は
「allowance を無視した場合」であり、$b_r$ に結果がどれだけ依存しているかを定量化する。
$\rho$ と $c_r$ は**スカラーとしてしか効かない**ので 1 つの job 内で全通りを評価する
（v1 は $c_r$ の 4 通りを 4 つの別 job にして fit を 4 重に無駄打ちしていた）。
`rff` では $b_r=1/\sqrt{n_{\mathrm{eval}}}$ とし $\rho$ に $c_r$ の役割を持たせる。

### 9.2. Reference drift の監査

各 reference について $B_r$ を §10 の解析式で計算し、$|B_r|\le b_r$ が**実際に成立したか**を
記録する（`reference_drift`, `allowance_covers_reference`）。v1 は $B_r$ を一度も計算して
いなかった。Main.tex L1131 の主張はこれで初めて裏づけられる。

### 9.3. 複数 reference と整合チェック

$\widehat U_a^{\mathcal R}=\min_{r}\left(|\widehat D_{a,r}|+q_{a,r}+b_r\right)$、
$\mathcal R=\{\texttt{correct},\texttt{misspecified}\}$、および

$$|\widehat D_{r,s}|\le q_{r,s}+b_r+b_s$$

の**違反率**を報告する。$q_{r,s}$ は**単一比較の正規半径**であり、候補族に対する同時半径では
ない（v1 の実装は候補半径を流用しており、閾値が一桁以上大きくなって検定が原理的に発火しな
かった）。この検定の実測された検出力については §18.3 を見よ。

---

## 10. Audit（解析的評価）

v1 は audit bias を `mean(m + α(y−γ)) − 1` で評価していた。これは outcome noise を含むため
MC 誤差が $\mathrm{sd}(\hat s)/\sqrt{n_{\mathrm{int}}}$ となり、$10^5$ 点でも $\approx0.006$ である。

simulation では $\alpha_0,\gamma_0,\sigma^2$ が既知なので、**noise を含まない厳密式**を使う：

$$B_a=E_{\mathrm{int}}\left[(\alpha_0-\widehat\alpha_a)(\widehat\gamma-\gamma_0)\right],$$
$$V_a=\operatorname{Var}_{\mathrm{int}}\left[m_a+\widehat\alpha_a(\gamma_0-\widehat\gamma)\right]
+E_{\mathrm{int}}\left[\widehat\alpha_a^2\right]\sigma^2,\qquad \sigma^2=1,$$

$m_a=\widehat\gamma(1,Z)-\widehat\gamma(0,Z)$。第 1 式は Main.tex eq `candidate_bias` そのもの、
第 2 式は $E[\varepsilon|X]=0$ から交差項が消えることによる。両式が実際に score の
モーメントを再現することを unit test で確認してある。

integration 標本は **(design, overlap, $b$) ごとに固定 seed で 1 回だけ生成**し全 replication で
共有する（v1 は replication ごとに新規生成）。$Y$ は不要なので生成しない。
サイズは low 100,000 / high 50,000、25,000 行単位で chunk 処理する。

---

## 11. Diagnostics と誤差確率の配分

v1 は mean radius・variance bound・reference 楕円体にそれぞれ $\delta/(2K)$ を割り当て、
合計が $1.5\delta$ になっていた。

**修正の要点は、variance bound は被覆の主張に入らないことである。**
Thm `uniform_selected_inference` が要求するのは $|B_a|\le\widehat U_a$ だけであり、
$\widehat U_a=|\widehat D_a|+q_a+b_r$ に $\widehat V_a^+$ は現れない。$\widehat V_a^+$ は
Thm `nested_oracle` の risk の主張にしか使われない。

| 事象 | 記号 | 配分 |
|---|---|---|
| 全体の miscoverage | $\tau$ | 0.05 |
| 同時 bias bound（被覆に効く） | $\delta$ | 0.01 |
| — fold あたり | $\delta/K$ | 0.002 |
| — うち mean radius $q_a$ | $\delta/(2K)$ | 0.001 |
| — うち reference allowance $b_r$ | $\delta/(2K)$ | 0.001 |
| — — $r_\alpha$ 楕円体 / $r_\gamma$ 楕円体 | $\delta/(4K)$ ずつ | 0.0005 |
| evaluation の正規近似 | $\tau-\delta$ | 0.04 |
| variance bound（risk の主張のみ・被覆とは独立） | $\delta_V$ | 0.01 |

`DeltaBudget.bias_budget_is_exhausted()` が配分の一致を保証し、unit test で固定してある。

---

## 12. Inference

| 区間 | 定理 | 扱い |
|---|---|---|
| `wald_split` / `wald_cf` | Cor `selected_wald` | 通常の Wald |
| `bias_aware_split` | Thm `uniform_selected_inference` | **定理あり**。fold 0 単独。定理の直接検証 |
| `conservative_cf` | eq `crossfit_bias_aware_half_length` | **定理あり**。cross-fit の主役 |
| `bias_aware_pooled` | **なし** | 計算はするが「理論的裏づけなし」と明示して報告 |

v1 は 5 fold を連結した pooled se に単一分割用の critical value を当てており、これを裏づける
定理は原稿にない。v2 では `bias_aware_split` と `conservative_cf` を主表に据え、
`bias_aware_pooled` は参考値に落とす。3 者の長さの差を見れば、cross-fit 版の定理を
書く価値があるかも判断できる（§19 の判断待ち事項）。

---

## 13. 計算計画

### 13.1. Tier

| Tier | grid A / B / C の replications | 目安 |
|---|---|---|
| `smoke` | 2 / 2 / 2 | 1 分未満 |
| `pilot` | 25 / 50 / 25 | 約 1 コア時間 |
| `publication` | 1,000 / 2,000 / 500 | 約 94 コア時間 |

v1 は「fast mode を作らない」方針だったが、これは replication package の要件に反する。
**tier は replication 数のみを変え、候補集合・選択規則・DGP・seed 設計は一切変えない。**

### 13.2. Publication grid

| Grid | design | 設定 | R | jobs |
|---|---|---|---|---|
| A（overlap 掃引） | low | $n\in\{1000,3000\}$，$s\in\{0.5,1.5,2.5\}$，$t=0$ | 1,000 | 6,000 |
| B（bias 掃引・**E4 主表**） | low | $n\in\{1000,3000\}$，$s=1.5$，$t\in\{0.5,1,2,4\}$ | 2,000 | 16,000 |
| C | high | $n=3000$，$s\in\{0.75,2.0\}$，$t\in\{0,1\}$ | 500 | 2,000 |

overlap と bias を全交差させないのは、$s$ が weight の裾を、$t$ が bias を動かす別々の軸であり、
交差項に主張がないためである。B に replication を厚く配るのは、そこが一様被覆の主張を
担うからである（被覆の MCSE は B で 0.0049、A で 0.0069、C で 0.0097）。
全 reference 種別・全 $\rho$・全 $c_r$・全選択規則は**各 job の内部で**評価される。

**実測コスト**（1 コア、1 replication）: low $n=1000$ 7.6 s / low $n=3000$ 12.7 s /
high $n=3000$ 57.5 s。合計 $\approx$ 17 + 45 + 32 = **94 コア時間**。
grid C の replication を 500 に落としているのは、1 replication が最も高価で、
かつ高次元設計が副次的な確認だからである（被覆の MCSE は 0.0097）。

### 13.3. 出力

`notebooks/experiments/results/reference_selection/<tier>/` に batch 単位の Parquet。
**`.gitignore` に追加済み**（v1 は未追加で、本実行すると 50MB 制限と git を直撃した）。

| File | 単位 |
|---|---|
| `candidate_*.parquet` | 候補 × fold |
| `selection_*.parquet` | (規則, reference, $\rho$) × fold |
| `repetition_*.parquet` | (規則, reference, $\rho$) × replication |
| `bound_*.parquet` | (reference, $\rho$) × fold |
| `check_*.parquet` | reference 対 × fold |
| `oracle_*.parquet` | 規則 × fold |

---

## 14. 報告基準

- 被覆・選択頻度には必ず MCSE $\sqrt{\hat p(1-\hat p)/R}$ を付す（最悪ケース行にも）。
- **single-split 区間は fold 0 の可用性だけで判定する。** Thm `uniform_selected_inference` は
  単一分割の主張なので、無関係な fold の失敗を被覆失敗として数えてはならない。
  cross-fit 区間（`wald_cf`・`conservative_cf`・`bias_aware_pooled`）は全 fold の完了を要する。
- **選択規則は (rule, reference, $\rho$) で区別する。** 同じ `proposed` でも reference が違えば
  別の手続きであり、平均してはならない（MCSE の分母も合わなくなる）。
- **一様性の主張には DGP 族上の最小被覆を主表に出す。** 平均被覆は補助。
- **被覆は無条件**とする。推定量を出せなかった replication は分母に残し「被覆せず」と数える。
  条件付き被覆だけを報告すると失敗の多い規則が不当に良く見える（`fixed_bkl` は ATE で
  常に失敗する）。この規約は `report._coverage_frame` に実装してあり notebook 側の裁量にしない。
- E2 では定理の**上側** $\widehat U_a\le|B_a|+2(q_a+b_r)$ の成立率も報告する。
- E2 では $\widehat U_a$ を $(|\widehat D_a|,q_a,b_r)$ に分解し、どの項が binding かを示す。

### 14.1. 主表の形

**Table E4（一様被覆）** — `uniform_coverage_table` が $t$ ごとの掃引、
`worst_case_coverage_table` が headline の最悪ケース行を出す：

| interval | min 被覆 | MCSE | 達成した $t$ | その $t$ での長さ | 定理あり |
|---|---|---|---|---|---|
| wald_split | | | | | ✓ |
| bias_aware_split | | | | | ✓ |
| conservative_cf | | | | | ✓ |
| bias_aware_pooled | | | | | **なし** |

最悪ケース行は列ごとの min ではなく **`idxmin` で選んだ 1 行**から取る。列ごとに min を取ると
被覆と長さが別のシナリオ由来になり、MCSE も付かない。

Wald は $2\Phi(1.96-t)-1$ に沿って崩れるはずであり（図に理論曲線を重ねる）、
bias-aware は $t$ によらず $\ge0.95$ を保つはずである。**この対比が論文の核心図表になる。**

**Table E1b（horse race）** — 行が選択規則、列が RMSE・regret・被覆・長さ。

---

## 15. テスト

`tests/test_reference_selection_experiment.py`（**`tests/` に置く** — v1 は
`notebooks/experiments/` にあり `make test` の対象外だった）。42 件。

| 分類 | 内容 |
|---|---|
| DGP | 任意の $b$ で ATE=1／fold rotation の disjoint 性と各観測がちょうど 1 回 evaluation／seed の順序非依存 |
| **E1a** | **再スケールで $\widehat\alpha$ 不変・目的値は $\kappa$ 倍**（SQ・UKL） |
| **候補集合** | **BP(1) の曲率が SQ と一致**／90 ラベルが相異なり BP(1) を含まない |
| **E4 の前提** | **実際に使う空間での非対称性**（candidate rich $R^2<0.5$、`correct` reference の outcome/propensity $>0.999$、`misspecified` $<0.5$） |
| dictionary | 標準化が training のみで決まる／SQ 候補の符号／**batched `alpha_matrix` が個別 `predict_alpha` と一致** |
| **効率不変条件** | **basis 3 個・generator 5 個の共有** |
| **audit** | **解析式が noisy Monte Carlo 平均と一致**／truth reference の bias が厳密に 0 |
| **reference** | **`correct` は $b\in\{0,1\}$ で allowance を守る**／**`misspecified` は $b=1$ で破る**／truth の allowance は 0 |
| **予算** | **bias 事象の合計がちょうど $\delta$**（v1 の配分なら $1.5\delta$ になることも確認）／`Numerics.n_folds` と `DeltaBudget.n_folds` の不一致を拒否 |
| inference | bounded-normal-mean が単調・公称被覆を厳密に達成／min-bound が各 bound 以下 |
| **再現性** | **全 6 テーブルが出る**／同一 job は同一出力／batch 書き出しと再読込 |
| **失敗の可視性** | **設定した procedure は成否によらず必ず行が出る**（`fixed_bkl` を含む）／**単一分割区間は他 fold の失敗で無効化されない** |
| **来歴** | **manifest 無しの batch を拒否**／**範囲外 batch を拒否**／**digest が展開後の job 列を区別** |
| **reference の失敗** | **失敗した reference は selection・bounds・check から除外**／非有限な check は undecidable |

`make lint` の対象に `notebooks/experiments/refsel` を追加した（v1 は `src tests tools` のみで
実験コードが lint されていなかった）。

---

## 16. v1 からの変更点

| # | 変更 | 理由 |
|---|---|---|
| 1 | E1a・E1b（再スケール不変性と horse race）を新設 | §4 の存在意義が未検証だった（R1） |
| 2 | drifting bias 族と $t$ 校正を新設、最小被覆を主表化 | 一様性の主張に最悪ケースがなかった（R2） |
| 3 | `misspecified` reference・$\rho$ 掃引・reference check を新設 | $b_r$ への依存が未検証だった（R3） |
| 4 | **reference に自前の outcome estimator を持たせた** | 共有だと reference 自身が allowance を破る（§18.1） |
| 5 | audit を解析式に変更（noise なし）、$B_r$ も監査 | MC 誤差が $B$ と同オーダーで被覆判定が信用できなかった |
| 6 | integration 標本を scenario 単位で共有・特徴量をキャッシュ | 計算量の 8 割が audit だった |
| 7 | **generator を $(\text{loss},\omega)$ 単位で共有し branch cache を導入** | branch 選択子が 1 replication で 1,370 万回呼ばれていた（§18.4） |
| 8 | $c_r$・$\rho$・reference 種別・選択規則を 1 job 内で評価 | fit の 4 重無駄打ちを解消 |
| 9 | $\delta$ 予算を再配分（variance を分離） | 合計が $1.5\delta$ で宣言値を超えていた |
| 10 | `bias_aware_split` を主役にし pooled は「理論なし」と明示 | cross-fit 版に定理がなかった |
| 11 | 3 tier 構成 | 40 コア日・fast mode なしでは replication package にならない（R6） |
| 12 | tests を `tests/` へ移動し 42 件に拡充、lint 対象に追加 | 品質ゲートを素通りしていた |
| 13 | `results/` を `.gitignore` へ | 本実行で 50MB 制限に抵触する |
| 14 | 選択後の再 fit を廃止（fold 内の結果を再利用） | 同一計算の二度打ち |
| 15 | NaN の `clip_binding_rate` を fail-closed に（clip を持たない generator のみ例外） | 監査 v3 の K-05/N-19 と同型の fail-open |

---

## 17. モジュール構成

```text
notebooks/experiments/refsel/
  dgp.py          DGP・drifting 族・fold rotation・seed
  candidates.py   basis・候補グリッド・ScaledGenerator・FoldLibrary（fit 共有）
  reference.py    4 種の reference・allowance・pairwise check
  selection.py    multiplier bootstrap・DeltaBudget・全選択規則
  audit.py        解析的 audit（integration 標本と特徴量のキャッシュ）
  inference.py    4 種の区間・MCSE
  calibration.py  $t$ 校正と calibration.json の読み書き
  grids.py        publication / smoke grid、tier 設定
  report.py       E1b–E6 の表
  rescaling.py    E1a
  calibration.json  コミット済みの校正表
```

---

## 18. 実装後に判明した事実（すべて実測）

計画段階の想定と食い違い、設計変更につながった事項。

### 18.1. 「全員が表現できない方向」は reference も壊す

当初計画では $\psi$ を candidate・reference の双方が表現できない方向としていた。
$n=3000$, $b=1$ での実測：

| reference | $B_r$ | $b_r$ | $\lvert B_r\rvert\le b_r$ |
|---|---|---|---|
| correct（当時：outcome 共有） | +0.649 | 0.111 | **偽** |
| misspecified（当時） | +0.584 | 0.068 | **偽** |

Thm `data_dependent_bias` の前提が壊れるので $t>0$ の測定が無意味になる。
§6.2.1 の非対称性を導入して修正した。修正後（$b=1$）：

| reference | $B_r$ | $b_r$ | 判定 | 候補 bound の被覆 |
|---|---|---|---|---|
| truth | 0.000 | 0 | 真 | 0.994 |
| correct | +0.0006 | 0.934 | **真** | 0.996 |
| misspecified | +0.587 | 0.119 | **偽** | **0.126** |

`misspecified` では候補 bias bound の被覆が 0.126 まで崩壊する。これが E5 の測定対象である。

**Prop `several_references` についての含意**: `min` bound は
無効な reference を 1 本混ぜるだけで被覆が 0.126 まで落ちる（無効な方の bound が小さいので
min に選ばれる）。命題は「各 $r$ について $|B_r|\le b_r$」を仮定しているので数学的には正しいが、
**実務上は min を取る操作が最も弱い reference の妥当性に完全に依存する**。
原稿で注意喚起する価値がある。

### 18.2. BKL は ATE で 100% 失敗する

$n=3000$, $s=1.5$, 5 fold・全 dictionary で BKL は 90/90 が `domain_error`。
Main.tex Table 5（既存の二標本実験）の「BKL failures 200」と整合し、
Main.tex の guidance 表が BKL を OWATE に割り当てていることとも整合する。
候補集合には残す（admissibility screen が非互換ペアを排除することの実演になる）が、
`fixed_bkl` は常に利用不能となるので、被覆を無条件で報告する規約（§14）が必須になる。

その他の失敗率（$n=3000$, $s=1.5$、全 450 fit 中）: `converged` 213、
`diagnostic_failure` 147、`domain_error` 90。admissible は約 50%。BP は rich/second-order で
30 中 25 が失敗する。これらは報告対象であって不具合ではない。

### 18.3. reference check の検出力は低い

$q_{r,s}$ を単一比較の正規半径に直した後でも、`correct` vs `misspecified` の違反率は
**5%**（$n=3000$, $b=1$, 20 fold）。$|\widehat D_{r,s}|=0.62$ に対し閾値
$q_{r,s}+b_r+b_s=1.77$ で、**有効な側の reference の allowance $b_r=0.93$ が閾値を支配する**。
すなわち、片方の reference の allowance が大きいと、他方が無効でも検定は発火しにくい。
eq `reference_check` の実用上の限界として報告すべきである。

### 18.4. 計算量のボトルネックは fit ではなく branch 選択子だった

プロファイル（low $n=3000$、1 replication）で `ate_branch` が **1,370 万回**呼ばれ、
全体 18.6 s のうち 7.1 s を占めていた。generator が候補ごとに 90 個作られており、
branch 符号のキャッシュ（`generators.py:283`）が候補間で共有されなかったためである。
$(\text{loss},\omega)$ 単位で 5 個に共有し、評価ブロックを `branch_caches()` で包んで
13.5 s → 9.9 s に短縮した。残りは L-BFGS そのもので、これ以上は削れない。

高次元は当初「1 replication が 40 分超」と誤って測定したが、これは macOS に `timeout`
コマンドが無く `|| echo` が誤発火しただけだった。実測は **44 秒**である。

### 18.5. bias bound は有益だが安くはない

$n=3000$, $t=0$, `correct` reference で $\widehat U/\widehat{se}=3.02$、
bias-aware 区間は Wald の 2.4 倍の長さ。`truth` reference では 0.96 で 1.4 倍。
$\widehat U$ の内訳は $q_a\approx0.41$、$b_r\approx0.24$ で、
**同時半径 $q_a$ の方が allowance より大きい**。候補数 90 に対する同時性の代償である。
Cor `oa:bias_aware_length` の前提 $\widehat U/\widehat{se}\to0$ はこの設計では成立しない。
E6 はこれを正直に報告する。

---

### 18.6. E4 は設計どおり動く（pilot 実測、$n=3000$, $R=120$）

校正した $t$ に対する被覆（`proposed` / `correct` / $\rho=1$）:

| $t$ | Wald 被覆 | 理論値 $2\Phi(1.96-t)-1$ | BA(split) 被覆 | Conservative 被覆 | BA/Wald 長さ比 | $\widehat U/\widehat{se}$ |
|---|---|---|---|---|---|---|
| 0 | 0.983 | 0.95 | 1.000 | 1.000 | 2.94 | 4.00 |
| 1 | 0.858 | 0.83 | 1.000 | 1.000 | 3.24 | 4.59 |
| 2 | 0.500 | 0.48 | 1.000 | 1.000 | 3.95 | 5.98 |
| 4 | 0.042 | 0.02 | 1.000 | 1.000 | 6.48 | 10.96 |

Wald は理論曲線にほぼ一致して崩れ、bias-aware と conservative は保つ。
**校正機構が正しく効いていることの確認になっている。**

### 18.7. Bias-aware 区間は保守的すぎる（最重要の未解決事項）

上表のとおり bias-aware の被覆は $t$ によらず 1.000 であり、公称 0.95 に対して過剰である。
$t=0$ ですら $\widehat U/\widehat{se}=4.0$、区間長は Wald の 2.9 倍。$t=4$ では 6.5 倍になる。

支配項は同時半径 $q_a$ である（§18.5）。90 候補（admissible は約 45）に対して
fold あたり $\delta/(2K)=0.001$ の同時性を要求するため、$q_a$ が $\widehat{se}$ と同オーダーで
残り、$n$ を増やしても比は縮まない（両方 $n^{-1/2}$）。

**査読者は必ず「常に被覆するのは区間が 3〜6 倍広いからだろう」と言う。** 対応の選択肢は
(a) 候補集合を絞る、(b) 同時性の補正を弱める（例: 候補を事前にグループ化して
group-wise に同時性を取る）、(c) 保守性を正直に報告し、Wald が壊れる領域で
**有効な**区間が他に無いことを主張する。

**〔2026-07-20 の pilot で (a) は却下された〕** §18.9 の weight screen で候補を
42.7 → 36.1 に絞っても（$t=0$、15% 減）、$q_a$ の中央値は 0.2485 → 0.2431 と 2% しか
縮まず、$\widehat U/\widehat{se}$ は 3.32 → **3.61 と悪化**、区間長比は 2.59 → 2.74 に増えた。
screen は分散の大きい候補を落とすため Wald 長さが 0.208 → 0.197 と縮み、$\widehat U$ が
ほぼ不変なので比が上がる。構造的な理由があり、$q_a$ は $\sqrt{\log p/n_{\mathrm{diag}}}$
オーダーなので候補数 $p$ を減らしても $\sqrt{\log}$ でしか効かない。
**候補集合の削減でこの保守性は解消しない。** 残る選択肢は (b)・(c) と、
$n_{\mathrm{diag}}$ を増やす fold 構成の変更（未実測、§8 の設計変更になる）である。
**著者の判断が必要。**

### 18.8. 提案規則は裾の重い候補を取り逃すことがある

$t=4$ での oracle regret 分布（fold 単位、$R=120$）:

| rule | median | p90 | p99 | max |
|---|---|---|---|---|
| `proposed` | 0.0093 | 0.097 | **94.2** | **2.5e3** |
| `score_var` | 0.0021 | 0.022 | 0.035 | 0.045 |
| `bregman_cv` | 0.0515 | 0.068 | 0.084 | 0.104 |

中央値では `proposed` が `bregman_cv` を明確に上回るが、**約 1% の fold で破滅的な候補
（`UKL|linear|c=0` などの無正則化 UKL）を選ぶ**。診断標本に極端な weight が現れなかった
場合、$|\widehat D_a|$ も $q_a$ も $\widehat V_a^+$ も小さく見積もられるためである。

これは手法の実質的な限界であり、Main.tex §4 の admissibility の段落
（"Weight concentration and inverse curvature can impose additional pre-specified
restrictions"）が想定している事態そのものである。**現在の実装はその追加制約を
入れていない。** weight 集中の事前制約を admissibility に加えるかどうかは
**著者の判断が必要**であり、加えるなら E1b の horse race に
「制約あり proposed」を 1 行足すのが自然である。

なお `n_ranked` は全規則で一致した（平均 45.2、範囲 40--50）ので、
規則ごとに候補集合が違うことによる交絡は $n=3000$ では生じていない。

### 18.9. weight screen は §18.8 を解消するが §18.7 には無力（pilot 実測、2026-07-20）

§18.8 の追加制約を `Numerics.min_ess_ratio` として実装した。候補 $a$ の representer が
診断 fold で持つ Kish 有効標本サイズ比
$\mathrm{ESS}_a/n_{\mathrm{diag}}=(\sum_i|\widehat\alpha_a(Z_i)|)^2/(n_{\mathrm{diag}}\sum_i\widehat\alpha_a(Z_i)^2)$
が閾値未満の候補を inadmissible にする。**weight を cap しないので estimand は変わらない**
（Main.tex L1207 が警告するのは cap の方であり、候補の screen は L734 が明示的に許容する）。
比はスケール不変なので E1a の再スケール不変性とも干渉しない。

閾値 4 条件 $\times$ $t\in\{0,4\}$ $\times$ $R=120$ を、同一 seed・同一 fit の対照として実測した
（960 jobs、27.2 分、12 並列）。$t=4$・`proposed`・`correct`・$\rho=1$ の oracle regret（fold 単位、各 $n=600$）:

| screen | median | p90 | p99 | max | `n_ranked` | 選択候補の ESS 比 min |
|---|---|---|---|---|---|---|
| なし | 0.0094 | 0.096 | **94.4** | **2529** | 45.2 | **0.0027** |
| 0.25 | 0.0093 | 0.085 | 14.8 | 2529 | 43.6 | 0.252 |
| 0.40 | 0.0092 | 0.078 | 1.95 | 1963 | 42.6 | 0.402 |
| 0.55 | 0.0091 | 0.064 | **0.379** | **1.86** | 40.9 | 0.550 |

- 閾値 0.55 で p99 が 250 分の 1、max が 1360 分の 1 になる一方、**median は悪化しない**
  （$t=0$ でも median 0.0087 → 0.0063、p99 0.146 → 0.069 と改善する）。
- 破滅例の正体が特定できた。screen なしのとき `proposed` が選んだ候補の ESS 比は
  最小 **0.0027**、すなわち全 weight が実質 3 観測に集中した representer を選んでいた。
- 副作用は観測されない。`available` は全条件で 1.000、admissible の最小は 24（$t=0$, 0.55）で
  全滅する fold は無く、選択される候補の分布もほぼ不変（`SQ|linear|c=4` が 112 → 114 fold）。
- **一方 §18.7 の保守性には効かない**（上記 §18.7 の追記を参照）。

**未確定**: 採用の可否と閾値は**著者の判断待ち**。0.55 は実測値だが、事前指定としては
丸い 0.5 の方が正当化しやすい。0.40 → 0.55 で p99 が 1.95 → 0.379 と大きく動くため
補間はできず、0.5 を採るなら追試が要る（$t=4$ のみ 120 rep で約 4 分）。
実装のデフォルトは `None`（制約なし）であり、本計画書の他の節が記述する挙動は変わらない。

- 生データ・サマリ・再実行スクリプト: `notebooks/experiments/results/weight_screen_pilot/`
  （`.gitignore` 対象。`summary.txt` に全表、`weight_screen_pilot.py` に実行条件）

---

## 19. 原稿側で確定が必要な点

コードではなく Main.tex の問題。

1. **L1125「Tuning uses only the training sample」** — 〔2026-07-20 に原稿を修正して解決〕
   実装は outcome も high-dim reference の ridge もハイパラ固定で、CV していない
   （`candidates.py:276-296`、`runner.py:124-125`）。v2 でも固定を維持する（候補比較の
   交絡を避けるため）ので、原稿の当該文を「training sample のみで fit し、ハイパラは
   事前固定で CV しない」旨に書き改めた。なお同じ文の
   "gradient-boosted outcome regression" は実測すると実装と一致しており（high design は
   `HistGradientBoostingRegressor`）、修正不要である。
2. **L1131「realized reference drift を近似する」** — v2 の §9.2 で実装したので裏づけられる。
3. **L1133 のベンチマーク 4 種** — v2 の E1b が `fixed_*`・`bregman_cv`・`lsif_cv`・`oracle` を
   実装するので裏づけられる。
4. **Prop `several_references`** — v2 の §9.3 が実装する。ただし §18.1・§18.3 の限界を
   本文に書き添えるべきである。
5. **cross-fit の bias-aware 区間** — 〔2026-07-20 に原稿を実測して解決〕
   `conservative_cf` に**裏づけはある**。Main.tex L866-880 が
   $\widehat H^{\mathrm{CF}}=\sum_k w_k(\widehat U_k+z_{1-(\tau-\delta)/(2K)}\widehat{se}_k)$ を定義し、
   直後に union bound による漸近被覆 $\ge 1-\tau$ を主張する。証明は OA L1354 にあり、
   fold ごとに Thm `uniform_selected_inference` の議論を normal-tail $(\tau-\delta)/K$・
   bias 失敗確率 $\delta/K$ で適用し、fold 事象の共通部分上で結論する。
   裏づけが無いのは `bias_aware_pooled`（pooled se に単一分割の critical value を当てる
   v1 の設計）だけであり、§12 のとおり v2 は既にこれを参考値に落としている。
   **残る判断は体裁のみ**: 本文の主張を `\begin{theorem}` 環境に格上げするかどうか。
6. **§7.2 の記述全体** — 結果が出てから書き直す。数値のない段階で主張を先に書かないこと。
7. **bias-aware 区間の保守性**（§18.7）— 〔2026-07-20 に選択肢 (a) を実測で却下〕
   候補集合を絞っても解消しない（§18.9）。同時性補正を弱めるか、保守性を正直に報告するか、
   $n_{\mathrm{diag}}$ を増やす fold 構成に変えるか。**著者の判断待ち。**
8. **weight 集中の admissibility 制約**（§18.8）— 〔2026-07-20 に実装・実測済み〕
   `Numerics.min_ess_ratio` として実装し、閾値 0.55 で regret p99 が 250 分の 1 になり
   副作用は観測されなかった（§18.9）。**採用の可否と閾値は著者の判断待ち。**
   採用する場合は Main.tex L1131「We report large weights without capping them」も
   修正対象になる（cap はしないが screen はする、という記述に改める）。
