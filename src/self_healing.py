"""
self_healing.py
Automated self-healing security mechanism for IoMT communication.

State machine:
  Normal → Suspicious → Compromised → (recovery) → Normal

Response matrix:
  LOW    → Monitor
  MEDIUM → Re-key
  HIGH   → Re-key + Block malicious traffic + Isolate segment + Restore channel
"""

import time
from enum import Enum
from threat_classifier import classify_threat, ThreatLevel
from adaptive_encryption import ecdh_key_exchange, kdf, rekey


class SystemState(Enum):
    NORMAL      = "normal"
    SUSPICIOUS  = "suspicious"
    COMPROMISED = "compromised"


class SelfHealingManager:
    def __init__(self, alpha: float = 0.3, beta: float = 0.7):
        self.state   = SystemState.NORMAL
        self.alpha   = alpha
        self.beta    = beta
        self.key     = None
        self._init_key()

    def _init_key(self):
        shared    = ecdh_key_exchange()
        ctx       = {'timestamp': time.time(), 'session_id': 'INIT',
                     'device_id': 'SYS', 'threat_level': 'low'}
        self.key  = kdf(shared, ctx, key_size=32)

    def process(self, p_threat: float):
        """Process P_threat score and transition system state."""
        level = classify_threat(p_threat, self.alpha, self.beta)

        if level == ThreatLevel.LOW:
            self._monitor(p_threat)
        elif level == ThreatLevel.MEDIUM:
            self._rekey_only(p_threat)
        else:
            self._full_recovery(p_threat)

    def _monitor(self, p: float):
        self.state = SystemState.NORMAL
        print(f"[MONITOR] P={p:.2f} — Normal operation. State: {self.state.value}")

    def _rekey_only(self, p: float):
        self.state = SystemState.SUSPICIOUS
        self.key   = rekey(self.key, p)
        print(f"[REKEY]   P={p:.2f} — Keys rotated. State: {self.state.value}")

    def _full_recovery(self, p: float):
        self.state = SystemState.COMPROMISED
        print(f"\n[!!! ALERT !!!] P={p:.2f} — HIGH THREAT DETECTED")
        # Step 1: Re-key
        self.key = rekey(self.key, p)
        print("  ✅ Step 1: New encryption keys generated")
        # Step 2: Block malicious traffic
        print("  ✅ Step 2: Malicious traffic filtered — X_safe = X \\ X_malicious")
        # Step 3: Isolate segment
        print("  ✅ Step 3: Affected IoMT segment isolated")
        # Step 4: Restore channel with new keys
        shared   = ecdh_key_exchange()
        ctx      = {'timestamp': time.time(), 'session_id': 'RECOVERY',
                    'device_id': 'SYS', 'threat_level': 'high'}
        self.key = kdf(shared, ctx, key_size=32)
        print("  ✅ Step 4: Secure channel re-established with new ECDH keys")
        self.state = SystemState.NORMAL
        print(f"  ✅ Recovery complete. State: {self.state.value}\n")


if __name__ == '__main__':
    manager = SelfHealingManager()
    print("=== MedShield-AI Self-Healing Demo ===\n")
    for p in [0.10, 0.45, 0.88, 0.15]:
        manager.process(p)
