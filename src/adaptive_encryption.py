"""
adaptive_encryption.py
Adaptive AES encryption with ECDH key exchange and dynamic re-keying.

Key exchange:  ECDH (X25519) — shared secret never transmitted over network
Key binding:   KDF(K, {timestamp, session_id, device_id, threat_level})
Encryption:    AES-128 (low threat) | AES-256 (medium/high threat)
Re-keying:     K_new = KDF(K_old, P_threat, t)
"""

import os, time, hashlib, hmac as hmac_lib
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from threat_classifier import classify_threat, ThreatLevel


def ecdh_key_exchange():
    """
    Elliptic Curve Diffie-Hellman using X25519.
    Q_A = d_A * G,  Q_B = d_B * G
    K   = d_A * Q_B = d_B * Q_A  (shared secret)
    """
    priv_a = X25519PrivateKey.generate()
    priv_b = X25519PrivateKey.generate()
    shared_a = priv_a.exchange(priv_b.public_key())
    shared_b = priv_b.exchange(priv_a.public_key())
    assert shared_a == shared_b
    return shared_a                       # never transmitted


def kdf(shared_secret: bytes, context: dict, key_size: int = 32) -> bytes:
    """
    Key Derivation Function with contextual binding.
    K_enc = KDF(K, [timestamp | session_id | device_id | threat_level])
    Ensures each session uses a unique key → forward secrecy.
    """
    ctx = (f"{context.get('timestamp','')}"
           f"|{context.get('session_id','')}"
           f"|{context.get('device_id','')}"
           f"|{context.get('threat_level','')}")
    return hmac_lib.new(
        shared_secret, ctx.encode(), hashlib.sha256
    ).digest()[:key_size]


def get_key_size(p_threat: float) -> int:
    """AES-128 for low threat, AES-256 for medium/high."""
    level = classify_threat(p_threat)
    return 16 if level == ThreatLevel.LOW else 32     # bytes (128 or 256 bits)


def aes_encrypt(key: bytes, plaintext: bytes) -> tuple:
    """AES-GCM authenticated encryption."""
    iv     = os.urandom(12)
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
    enc    = cipher.encryptor()
    ct     = enc.update(plaintext) + enc.finalize()
    return iv, ct, enc.tag


def aes_decrypt(key: bytes, iv: bytes, ct: bytes, tag: bytes) -> bytes:
    """AES-GCM authenticated decryption."""
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
    dec    = cipher.decryptor()
    return dec.update(ct) + dec.finalize()


def rekey(old_key: bytes, p_threat: float) -> bytes:
    """K_new = KDF(K_old, P_threat, t) — dynamic re-keying."""
    material = str(time.time()).encode() + str(p_threat).encode()
    return hmac_lib.new(old_key, material, hashlib.sha256).digest()[:len(old_key)]


if __name__ == '__main__':
    # Demo: ECDH → KDF → AES-256 encrypt → decrypt
    shared  = ecdh_key_exchange()
    ctx     = {'timestamp': time.time(), 'session_id': 'S001',
               'device_id': 'ECG_Monitor_01', 'threat_level': 'high'}
    key     = kdf(shared, ctx, key_size=32)       # AES-256
    message = b"Patient ECG data — CONFIDENTIAL"
    iv, ct, tag = aes_encrypt(key, message)
    pt      = aes_decrypt(key, iv, ct, tag)

    print(f"Original : {message.decode()}")
    print(f"Encrypted: {ct.hex()[:40]}...")
    print(f"Decrypted: {pt.decode()}")

    # Re-key under high threat
    new_key = rekey(key, p_threat=0.92)
    print(f"\nOld key: {key.hex()[:20]}...")
    print(f"New key: {new_key.hex()[:20]}...")
