"""
项目全局配置 —— 随机种子、交叉验证参数等。

所有训练 / 推理脚本统一从此处读取种子，确保划分可复现。
"""

# ── 各任务的 repeat → seed 映射 ──────────────────────────────────
# 训练时使用 enumerate(SEEDS) 遍历，repeat 即为列表下标。
# 推理时通过 --repeat 参数反查对应 seed。

# 诊断模型（train_diag.py）
DIAG_REPEAT_SEEDS = [20, 42]

# HC 情绪模型（train_hc.py）
HC_REPEAT_SEEDS = [20,  42]

# DEP 情绪模型（train_dep.py）
DEP_REPEAT_SEEDS = [20,  42]

# ── 交叉验证 ─────────────────────────────────────────────────────
N_SPLITS = 5          # K 折数
N_REPEATS = 2         # 默认 repeat 数（向后兼容）

# ── run_seed 构造规则 ────────────────────────────────────────────
# make_run_seed(base_seed, fold) = base_seed * 1000 + fold
RUN_SEED_MULTIPLIER = 1000


def make_run_seed(base_seed: int, fold: int, extra: int = 0) -> int:
    """为每个 seed × fold 构造一个独立但可复现的 run_seed。"""
    return int(base_seed) * RUN_SEED_MULTIPLIER + int(fold) + int(extra)


def get_seed_for_repeat(task: str, repeat: int) -> int:
    """根据任务名和 repeat 索引获取对应的 seed。"""
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
    if repeat < 0 or repeat >= len(seed_list):
        raise IndexError(
            f"repeat={repeat} out of range for task '{task}' "
            f"(seeds={seed_list}, valid range: 0-{len(seed_list)-1})"
        )
    return seed_list[repeat]
