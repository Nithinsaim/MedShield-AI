"""
threat_classifier.py
Threshold-based threat level classification from continuous P_threat score.

Classification:
  T(x) = Low    if P_threat < alpha
          Medium if alpha <= P_threat < beta
          High   if P_threat >= beta

Re-key interval adapts dynamically:
  t_update = 1 / (1 + P_threat)
"""

from enum import Enum
from dataclasses import dataclass


class ThreatLevel(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class SecurityConfig:
    aes_key_bits:   int     # 128 or 256
    rekey_interval: float   # seconds
    response:       str


POLICIES = {
    ThreatLevel.LOW:    SecurityConfig(128, 300.0, "monitor"),
    ThreatLevel.MEDIUM: SecurityConfig(256, 60.0,  "rekey"),
    ThreatLevel.HIGH:   SecurityConfig(256, 0.0,   "rekey + block + isolate"),
}


def classify_threat(p_threat: float,
                    alpha: float = 0.3,
                    beta: float  = 0.7) -> ThreatLevel:
    """Map P_threat ∈ [0,1] → discrete threat level."""
    if p_threat < alpha:
        return ThreatLevel.LOW
    elif p_threat < beta:
        return ThreatLevel.MEDIUM
    return ThreatLevel.HIGH


def get_rekey_interval(p_threat: float) -> float:
    """t_update = 1 / (1 + P_threat) — higher threat = more frequent re-keying."""
    return 1.0 / (1.0 + p_threat)


def get_policy(p_threat: float) -> SecurityConfig:
    level  = classify_threat(p_threat)
    config = POLICIES[level]
    config.rekey_interval = get_rekey_interval(p_threat)
    return config


if __name__ == '__main__':
    print(f"{'P_threat':<12} {'Level':<10} {'AES':<10} {'Rekey(s)':<12} {'Response'}")
    print("-" * 65)
    for p in [0.05, 0.20, 0.40, 0.60, 0.75, 0.95]:
        lvl = classify_threat(p)
        cfg = get_policy(p)
        print(f"{p:<12.2f} {lvl.value:<10} AES-{cfg.aes_key_bits:<6} "
              f"{cfg.rekey_interval:<12.3f} {cfg.response}")
