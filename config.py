"""
项目全局配置 —— 随机种子、交叉验证参数等。

所有训练 / 推理脚本统一从此处读取种子，确保划分可复现。
"""

# ── 各任务的 repeat → seed 映射 ──────────────────────────────────
# 训练时使用 enumerate(SEEDS) 遍历，repeat 即为列表下标。
# 推理时通过 --repeat 参数反查对应 seed。

# 测试集模型随机种子
TEST_SEED = [20,42]



# 诊断模型（train_diag.py）
# 可以按需扩展：例如加 123 支持第 3 次五折
DIAG_REPEAT_SEEDS = [20, 42]

# HC 情绪模型（train_hc.py）
HC_REPEAT_SEEDS = [20, 42]

# DEP 情绪模型（train_dep.py）
DEP_REPEAT_SEEDS = [20, 42]

# ── 交叉验证 ─────────────────────────────────────────────────────
N_SPLITS = 5          # K 折数
N_REPEATS = 2         # 默认 repeat 数（向后兼容）

# ── 验证推理模式 ─────────────────────────────────────────────────────
# True: 硬投票（窗口级多数表决 + 诊断硬路由）
#   - 每个窗口硬预测(>=0.5)，按被试/trial 多数表决
#   - 诊断结果硬路由到对应情绪模型（pred_diag=0→HC模型, 1→DEP模型）
# False: 软投票（概率均值 + 概率加权融合）
VAL_HARD_VOTING = True

# ── run_seed 构造规则 ────────────────────────────────────────────
# make_run_seed(base_seed, fold) = base_seed * 1000 + fold
RUN_SEED_MULTIPLIER = 1000


def make_run_seed(base_seed: int, fold: int, extra: int = 0) -> int:
    """为每个 seed × fold 构造一个独立但可复现的 run_seed。"""
    return int(base_seed) * RUN_SEED_MULTIPLIER + int(fold) + int(extra)


def get_seed_for_repeat(task: str, repeat: int) -> int:
    """根据任务名和 repeat 索引获取对应的 seed。

    如果 repeat 超出预定义列表，则用 base_seed + repeat * 31 派生一个新 seed，
    保证不同 repeat 有不同但可复现的种子。
    """
    seeds = {
        "diagnosis": DIAG_REPEAT_SEEDS,
        "diag": DIAG_REPEAT_SEEDS,
        "hc_emotion": HC_REPEAT_SEEDS,
        "hc": HC_REPEAT_SEEDS,
        "dep_emotion": DEP_REPEAT_SEEDS,
        "dep": DEP_REPEAT_SEEDS,
    }
    seed_list = seeds.get(task)
    if seed_list is None:
        raise KeyError(f"Unknown task: {task}. Expected one of {list(seeds.keys())}.")
    if repeat < 0:
        raise IndexError(f"repeat={repeat} 不能为负数。")
    if repeat < len(seed_list):
        return seed_list[repeat]
    # 派生种子：用列表中第一个种子 + 偏移，保持不同 repeat 可复现
    derived = seed_list[0] + repeat * 31
    print(f"[config] repeat={repeat} 超出预定义种子列表 {seed_list}，使用派生种子 {derived}")
    return derived
