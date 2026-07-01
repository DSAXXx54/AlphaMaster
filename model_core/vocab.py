from dataclasses import dataclass

from .ops import OPS_CONFIG


FEATURE_NAMES = (
    "RET",        # index 0
    "RET5",       # index 1
    "VOL_RATIO",  # index 2
    "PRESSURE",   # index 3
    "DEV",        # index 4
    "HL_RANGE",   # index 5
    "ATR",        # index 6
    "RVOL",       # index 7
    "RET20",      # index 8
    "AC1",        # index 9
)


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
