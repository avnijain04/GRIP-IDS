# src/crypto_utils.py
import os
import numpy as np
from io import BytesIO
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

def serialize_weights(weights_list) -> bytes:
    bio = BytesIO()
    np.save(bio, np.asarray(weights_list, dtype=object), allow_pickle=True)
    bio.seek(0)
    return bio.read()

def deserialize_weights(bytes_blob):
    bio = BytesIO(bytes_blob); bio.seek(0)
    arr = np.load(bio, allow_pickle=True)
    return [np.array(x, dtype=np.float32) for x in arr.tolist()]

def generate_aes_key():
    return AESGCM.generate_key(bit_length=256)

def aes_encrypt(key: bytes, plaintext: bytes):
    aesgcm = AESGCM(key); nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return nonce, ct

def aes_decrypt(key: bytes, nonce: bytes, ciphertext: bytes):
    aesgcm = AESGCM(key)
    pt = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return pt

def generate_ed25519_keypair():
    priv = Ed25519PrivateKey.generate(); pub = priv.public_key()
    return priv, pub

def sign_bytes(priv: Ed25519PrivateKey, data: bytes) -> bytes:
    return priv.sign(data)

def verify_signature(pub: Ed25519PublicKey, signature: bytes, data: bytes) -> bool:
    try:
        pub.verify(signature, data); return True
    except Exception:
        return False

def serialize_public_key(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

def deserialize_public_key(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)
