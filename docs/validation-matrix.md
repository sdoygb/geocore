# WCI/geoqc 验证覆盖矩阵（普适性盘点）

**日期**：2026-09-04（v1 盘点）；2026-09-04（v2：H₂S 盲测 + 两 bug 修复订正）
**目的**：回答"我们的方法能算几个分子、缺哪些维度"——把 geocore/geoqc 与文章线（10.87–10.91）的**实际验证证据**摊开，标出空白维度，供盲测验收与后续补测决策。
**口径纪律**："出现" ≠ "验证"。下表只记有对照/收敛/精度的实际运行入口（文件行可查），不记仅有名字的引用。**2026-09-04 重要订正**：v1 的"benchmark 对照"是假象——脚本曾因两 bug（见 E 节）从未真正跑通且 FCI 参考漏核能；修复后 4 系统真实收敛，并新增 H₂S 盲测 PASS。

---

## A. 验证入口清单（文件可查）

| 系统 | 基组 | 类型 | 验证内容 | 出处 | 质量等级 |
|---|---|---|---|---|---|
| H₂O | STO-3G | 闭壳 10e | WCI vs PySCF FCI（dim 441）| examples/geoqc_benchmark.py | ✅ 跑通 err 1.2e-12（2026-09-04 修后）|
| LiH | 6-31G | 闭壳 4e | WCI vs PySCF FCI（dim 3025）| examples/geoqc_benchmark.py | ✅ 跑通 err 3.3e-8（修后）|
| N₂ | STO-3G | 闭壳 14e | WCI vs PySCF FCI（dim 14400）| examples/geoqc_benchmark.py | ✅ 跑通 err 2.5e-4（修后）|
| CH₄ | STO-3G | 闭壳 10e | WCI vs PySCF FCI（dim 15876）| examples/geoqc_benchmark.py | ✅ 跑通 err 2.5e-5（修后）|
| **H₂S** | **STO-3G** | 闭壳 18e | **盲测**：WCI vs PySCF FCI（dim 3025，第三周期首样本，零调参）| examples/geoqc_h2s_blind.py | ✅ **PASS err 2.8e-8** |
| LiH | 6-31G | — | 积分布局逐元素校验（修过 −19.25 bug）| examples/geoqc_mkint.py | 数值层 |
| S_z 扇区 | — | — | 扇区构建 == 全 N 扇区（机器精度）| tests/test_geoqc_sz.py | 数值层 |
| Grassmann SCF | — | — | geoqc SCF == PySCF RHF（M2/M3）| tests/test_geoqc_scf.py | 数值层 |
| N₂ | STO-3G | 多参考解离 | 解离曲线 R=1.1→2.6：geoqc FCI == openfermion/PySCF FCI（平衡+解离两区）| examples/geoqc_n2.py, tests/test_geoqc_n2.py | 对照（含多参考区）|
| N₂ | 6-31G fc | 基组协变性 | canonical MO vs NO 基：波包稀疏性基组依赖（非协变结论）| examples/geoqc_no_basis.py | 方法学判别 |
| LiH / N₂ / H₂O | STO-3G | — | FCI 谱矩（免存储对角化验证）| examples/geoqc_moments.py | 数值层 |
| H₂O/CH₄/NH₃/BH₃ | STO-3G | 异核氢化物解离 | 谱聚集 ⟨r⟩ × 键长（100 点/分子）→ 10.90 三因子模型 | examples/geoqc_bondtype_r.py, _r2.py | 谱诊断（跨系统）|
| N₂/O₂(spin2)/C₂H₂ | STO-3G | 同核键解离 | ⟨r⟩ 曲线（O₂/C₂H₂ 30 点，C₂H₂ dim 627k）| examples/geoqc_bondtype_r2.py | 谱诊断 |
| H₂O | cc-pVDZ | 中规模 | WCI（max_wp=50，energy_tol=1e-4 Ha = 0.1 mHa）| examples/geoqc_h2o_ccpvdz.py | 收敛目标 0.1 mHa |
| H₂O | cc-pVDZ | GPU | WCI GPU/CPU 加速基准（5 波包）| examples/geoqc_h2o_gpu_bench.py | 性能 |
| **H₂CO** | **cc-pVDZ** | **dim=2.392×10¹⁵** | PT2 修正 + ball-cover 选择；block-Newton（10.91 §6）vs CCSD(T) | examples/wci_pt2_h2co_test.py, wci_pt2_h2co_light.py, wci_block_pt2_h2co_test.py | **10¹⁵ 验收线** |
| LiH | STO-3G/6-31G | M1 | 约束子流形 Newton 0.0002 mHa | geo10-15/solver（10.91）| 单点收敛 |

---

## B. 维度覆盖矩阵

维度列 × 现有证据：

| 维度 | 已覆盖 | 空白/薄弱 |
|---|---|---|
| **分子类型** | H₂O、LiH、N₂、CH₄、NH₃、BH₃、O₂、C₂H₂、H₂CO、HF、LiF?、C₂、CN、H₃ + **H₂S（第三周期首例，盲测）** | 第三周期其余（PH₃、SiH₄…）空；3d 过渡金属（FeO、MnO…）全空；离子对（NaCl…）空 |
| **开/闭壳** | 闭壳为主；O₂ 三重态（spin=2）单点 | 开壳自由基（CH₃、OH、NO）未测收敛 |
| **多参考强度** | N₂ 解离（教科书强多参考）、H₂O 断键、C₂H₂（10.90 深聚集 0.25）| 弱多参考区（平衡附近电荷转移如 LiH）只测了 ⟨r⟩，未测 WCI 收敛 |
| **键长区** | 平衡点为主；N₂ 1.1→2.6 Å、氢化物 100 点解离扫描 | 过渡态/势垒区；压缩区（r<r_eq）几乎未测 |
| **基组** | STO-3G、6-31G、cc-pVDZ | cc-pVTZ+、极化弥散、无基组极限外推 |
| **系统规模** | dim 441 → 3025 → 14400 → 15876 → 627k → H₂O/cc-pVDZ → **2.392×10¹⁵**（H₂CO）| 10¹⁵ 档只有 H₂CO 一个样本 |
| **性质** | 总能量（基态）、谱聚集诊断 | 梯度/几何优化、激发态、密度/偶极 未在 WCI 线验证 |
| **数值层** | 积分（LiH 6-31G 逐元素）、S_z 扇区、Grassmann SCF == RHF | cc-pVDZ 级以上积分的独立逐元素校验未覆盖 |

---

## C. 空白维度与风险（普适性要害）

1. **周期盲区最危险**：全部验证在第二周期及以内（H–F）。换到第三周期（S、P、Cl）与 3d 过渡金属，轨道结构、谱刚性、多参考形态都可能变——这正是"换分子就失效"最可能爆发的地方，却一个都没测。
2. **10¹⁵ 档单样本**：H₂CO/cc-pVDZ 是唯一 10¹⁵ 样本——收敛改进若只在它上调参，就是"几个分子有效"的典型形态。需要一个同规模盲测分子（如 C₂H₄/cc-pVDZ 或乙烷活性空间）做验收。
3. **⟨r⟩ 判据是当前普适性证据最硬的模块**（跨基组稳健，10.89/10.90）——它作为 WCI 波包自适应的控制信号，其普适性 = 方法普适性的地基；该主张本身还需在第三周期/3d 上检验。
4. **收敛质量**（v2 订正）：benchmark 四系统修复后真实收敛（err 1e-12–2.5e-4 Ha），但**仍非 CI 回归**——需固化为回归断言（D.1 依然成立）；H₂CO 的 PT2/Newton 结果精度记录在 examples 输出中，非 CI 回归。

---

## E. 2026-09-04：H₂S 盲测与两 bug 修复记录

**盲测（validation D.2 执行）**：H₂S/STO-3G（18e，dim 3025，第三周期首个从未参与开发的样本），结构/参数与 benchmark 完全相同（max_wp=30, tol=1e-8, FCI conv_tol=1e-10，零调参）。
**结果：PASS**——|E_WCI − E_FCI| = 2.8×10⁻⁸ Ha。→ 普适性第一弹：WCI 在未见过的第三周期分子上直接收敛。

**过程中修复的两个真 bug（v1 矩阵"已覆盖"区藏着的假象）**：

1. **wci/exterior src 索引崩溃（核心，全系统）**
   - 现象：`IndexError: boolean index ... 66048 vs 66097`——H₂O/LiH/N₂/CH₄/H₂S **全线崩溃**（benchmark 脚本此前从未真正跑通，v1"对照"为假象）。
   - 根因：wci.py `build_H_incremental` 等处以"每源输出目标数相同"（`len//n_chunk`）重建 src_idx；源间激发数不均即错位（H₂S 首 chunk 129 源、66097 目标 vs 等长假设 66048）。
   - 修复（`geoqc/exterior.py` + `geoqc/wci.py`）：`sparse_action_sz` 的 apply 返回**真实源索引** t_src（与目标同步逐段填充）4 元组；wci 走既有 `len==4` 分支。空返回分支同步 4 元组。
   - 回归：核心 30 测试 passed；benchmark 4 系统 err 1.2e-12 / 3.3e-8 / 2.5e-4 / 2.5e-5 Ha。
   - 遗留：`sparse_action_sz_vec`（GPU/向量化）与 `parallel_apply_factory` 的 apply 若仍 3 元组返回，同类等长假设风险仍在 vec/gpu 路径——未查，跑 GPU 基准前须先查。

2. **examples FCI 参考漏核排斥能（假 FAIL）**
   - 现象：修复 1 后全线"不收敛"，err 恰等于各分子 E_nuc（H₂O 9.19、N₂ 23.6、H₂S 12.95 Ha）。
   - 根因：PySCF FCI kernel 返回能量**不含核排斥**；benchmark/盲测脚本未加 `mol.energy_nuc()`，而 WCI exterior 已带 nuc——对比基准错位。
   - 修复：`run_pyscf_fci` 返回 `e + mol.energy_nuc()`。
   - 教训：v1 若当时真跑过 benchmark 就会立刻看到假 FAIL；"脚本存在"≠"验证通过"——已记入口径纪律。

**提交**：geocore b8f8907（fix+blind）。

---

## D. 建议（按序）

1. **固化回归**：把 A 表入口做成 CI/回归（至少四个 benchmark 系统的 WCI==FCI 断言），防止"修 A 坏 B"——这是多分子普适性的最低保障；
2. **盲测第一弹**：挑一个从未参与开发的第三周期分子（H₂S 或 SiH₄，STO-3G/6-31G），跑 WCI vs PySCF FCI——直接检验是否"换分子失效"；
3. **⟨r⟩ 扩展检验**：在第三周期/3d 小系统上验证谱聚集判据（5+ 系统、3 基组证据目前限第二周期）；
4. **10¹⁵ 盲测第二弹**：同规模新分子验收（确认收敛改进不是 H₂CO 专属）；
5. 每步失败要可归因（波包/截断/谱结构哪一环）——配套绝对误差走廊与收敛地图。
