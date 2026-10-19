"""集中配置；所有相对路径均以项目根目录解析，便于整体迁移。"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass
class DataConfig:
    panel_path: str = "code/data/data.csv"
    calendar_path: str = "code/data/trading_calendar.csv"
    industry_path: str = "code/data/stock_industry_sw2021.csv"
    industry_dictionary_path: str = "code/data/sw2021_l2_mapping.csv"
    cache_dir: str = "code/data/cache_static134"
    history_length: int = 60
    future_length: int = 20
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    missing_ratio_threshold: float = 0.2
    clip_quantiles: list[float] = field(default_factory=lambda: [0.001, 0.999])
    clip_reservoir_size: int = 200_000
    min_stocks: int = 2


@dataclass
class ModelConfig:
    name: str = "factorgcl"
    prior_factors: int = 134  # 用户确认的数据替代；原论文为 83。
    input_dim: int = 6
    embedding_dim: int = 32
    gru_layers: int = 2
    hidden_factors: int = 32
    projection_hidden_dim: int = 32
    projection_output_dim: int = 32
    negative_slope: float = 0.01  # 原文未给出，沿用指南默认。
    batchnorm_eps: float = 1e-5
    batchnorm_momentum: float = 0.1
    share_future_encoder: bool = False
    use_prior: bool = True
    use_hidden: bool = True
    use_alpha: bool = True
    use_cl: bool = True


@dataclass
class TrainingConfig:
    seed: int = 0
    learning_rate: float = 0.001
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 20
    min_delta: float = 0.0
    temperature: float = 0.1
    contrastive_weight: float = 0.1
    shuffle_dates: bool = True
    device: str = "auto"
    cpu_threads: int = 4
    contrastive_anchor_chunk: int = 512
    output_dir: str = "code/results/main"
    monitoring_enabled: bool = False
    monitoring_dates_per_epoch: int = 3
    scaler_reference_dir: str | None = None  # 复用经签名核对的同数据/同折训练裁剪，不是复用模型。


@dataclass
class FoldConfig:
    name: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    folds: list[FoldConfig] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def validate_config(cfg: Config):
    if cfg.model.prior_factors != 134:
        raise ValueError("当前代码使用用户确认的完整 SW2021 字典，prior_factors 必须为 134。")
    if cfg.model.name not in {"factorgcl", "gru"}:
        raise ValueError("model.name 仅支持 factorgcl 或同编码器 GRU 基线。")
    if cfg.model.input_dim != 6 or not cfg.data.horizons:
        raise ValueError("六价量特征或收益期限配置无效。")
    if cfg.model.hidden_factors < 0:
        raise ValueError("hidden_factors 不能为负。")
    if any(h < 1 for h in cfg.data.horizons) or len(set(cfg.data.horizons)) != len(cfg.data.horizons):
        raise ValueError("收益期限必须是互不重复的正整数。")
    if cfg.training.cpu_threads < 1 or cfg.training.learning_rate <= 0:
        raise ValueError("CPU 线程数和学习率必须为正。")
    if not any((cfg.model.use_prior, cfg.model.use_hidden and cfg.model.hidden_factors > 0, cfg.model.use_alpha)):
        raise ValueError("至少保留一个历史预测模块。")
    if cfg.model.use_cl and not cfg.model.use_alpha:
        raise ValueError("正文 wo Alpha&CL 同时关闭 alpha 和 CL；不能定义不存在的 alpha 对比。")
    if cfg.data.history_length < 2 or cfg.data.future_length < 1:
        raise ValueError("窗口长度无效。")
    if not 0 <= cfg.data.missing_ratio_threshold < 1:
        raise ValueError("missing_ratio_threshold 必须位于 [0,1)。")
    if not (0 <= cfg.data.clip_quantiles[0] < cfg.data.clip_quantiles[1] <= 1):
        raise ValueError("裁剪分位点无效。")
    if cfg.training.temperature <= 0 or cfg.training.max_epochs < 1 or cfg.training.patience < 1:
        raise ValueError("温度、训练轮数或 patience 无效。")
    if cfg.training.contrastive_weight < 0:
        raise ValueError("对比损失权重不能为负。")
    if cfg.training.monitoring_dates_per_epoch < 1:
        raise ValueError("训练监测日期数必须至少为 1。")
    if cfg.data.min_stocks < 2:
        raise ValueError("GRU 输出 BatchNorm 与相关评价至少需要两只股票。")
    if cfg.data.clip_reservoir_size < 1 or cfg.training.contrastive_anchor_chunk < 1:
        raise ValueError("分位点样本容量和 CL anchor 块大小必须为正。")
    for fold in cfg.folds:
        if not (
            fold.train_start <= fold.train_end < fold.valid_start <= fold.valid_end
            < fold.test_start <= fold.test_end
        ):
            raise ValueError(f"滚动折日期不满足时间顺序：{fold.name}")
    if len({fold.name for fold in cfg.folds}) != len(cfg.folds):
        raise ValueError("fold.name 不得重复。")
    ordered = sorted(cfg.folds, key=lambda x: x.test_start)
    for left, right in zip(ordered, ordered[1:]):
        if left.test_end >= right.test_start:
            raise ValueError("测试折重叠，不能唯一拼接样本外预测。")


def load_config(path: str | Path) -> Config:
    raw = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {"data", "model", "training", "folds", "notes"}
    if unknown:
        raise ValueError(f"未知配置节：{sorted(unknown)}")
    cfg = Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        folds=[FoldConfig(**x) for x in raw.get("folds", [])],
        notes=raw.get("notes", {}),
    )
    validate_config(cfg)
    return cfg
