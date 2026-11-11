"""Защита капитала от просадки (drawdown guard).

  * Просадка от пика >= soft_drawdown  → позиции плавно уменьшаются: чем ближе
    к жесткому лимиту, тем меньше риск (до soft_scale у самого лимита).
  * Просадка от пика >= hard_drawdown  → закрываем всё, пауза cooldown_bars баров.
    После паузы торговля возобновляется с урезанным риском: отсчет просадки
    идет так, будто бот уже потерял resume_drawdown (режим "pause"),
    либо бот останавливается до ручного вмешательства (режим "halt").
  * Режим "scale": жесткого стопа нет. Чем глубже просадка от исторического пика,
    тем меньше позиции (у hard_drawdown и глубже — soft_scale от обычного размера).
    Пик не сбрасывается, поэтому бот не может "забыть" о потерях.

Проверка жесткого лимита идет и внутри бара (в live — при каждом опросе цены),
поэтому фактический убыток обычно близок к лимиту, но при гэпах может быть больше.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import RiskConfig


@dataclass
class DrawdownGuard:
    cfg: RiskConfig
    peak: float
    paused_left: int = 0
    halted: bool = False
    events: list[dict] = field(default_factory=list)

    def drawdown(self, equity: float) -> float:
        return 1 - equity / self.peak if self.peak > 0 else 0.0

    @property
    def active(self) -> bool:
        """True, если сейчас торговля запрещена (пауза или остановка)."""
        return self.halted or self.paused_left > 0

    def check_intrabar(self, equity: float, ts=None) -> bool:
        """Проверка жесткого лимита. True — нужно немедленно закрыть всё."""
        if self.active or self.cfg.hard_stop_action == "scale":
            return False
        if self.drawdown(equity) >= self.cfg.hard_drawdown:
            self._trigger_hard(equity, ts)
            return True
        return False

    def on_bar_close(self, equity: float, ts=None) -> float:
        """Вызывается на закрытии бара. Возвращает множитель для целевых весов (0..1)."""
        if self.halted:
            return 0.0
        if self.paused_left > 0:
            self.paused_left -= 1
            if self.paused_left == 0:
                # стартуем "в просадке": полный размер позиций вернется только после
                # того, как бот заработает resume_drawdown обратно
                self.peak = equity / (1 - self.cfg.resume_drawdown)
                self.events.append({"ts": ts, "event": "resume", "equity": equity})
            return 0.0

        self.peak = max(self.peak, equity)
        dd = self.drawdown(equity)
        if self.cfg.hard_stop_action == "scale":
            dd = min(dd, self.cfg.hard_drawdown)
        elif dd >= self.cfg.hard_drawdown:
            self._trigger_hard(equity, ts)
            return 0.0
        if dd >= self.cfg.soft_drawdown:
            span = self.cfg.hard_drawdown - self.cfg.soft_drawdown
            k = (dd - self.cfg.soft_drawdown) / span if span > 0 else 1.0
            return 1.0 - k * (1.0 - self.cfg.soft_scale)
        return 1.0

    def _trigger_hard(self, equity: float, ts) -> None:
        if self.cfg.hard_stop_action == "halt":
            self.halted = True
        else:
            self.paused_left = self.cfg.cooldown_bars
        self.events.append(
            {"ts": ts, "event": "hard_stop", "equity": equity, "drawdown": self.drawdown(equity)}
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cfg")
        d["events"] = [{**e, "ts": str(e["ts"])} for e in self.events[-50:]]
        return d

    @classmethod
    def from_dict(cls, cfg: RiskConfig, d: dict) -> "DrawdownGuard":
        return cls(cfg=cfg, **d)
