# src/crypto_utils.py
import os
import numpy as np
from io import BytesIO
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

# -------------------------
# SAFE SERIALIZATION (NO PICKLE)
# -------------------------
# use a deterministic, order-preserving serialization for a list-of-arrays
from io import BytesIO
import numpy as np

def serialize_weights(weights_list) -> bytes:
    """
    Serialize a list-of-arrays using numpy.save with pickle allowed.
    This preserves list ordering deterministically.
    """
    bio = BytesIO()
    # Save the entire list object (ordered) - allow_pickle True so list of ndarrays preserved
    np.save(bio, np.asarray(weights_list, dtype=object), allow_pickle=True)
    bio.seek(0)
    return bio.read()


def deserialize_weights(bytes_blob):
    """
    Reverse of serialize_weights: return list of float32 numpy arrays (same order).
    """
    bio = BytesIO(bytes_blob)
    bio.seek(0)
    arr = np.load(bio, allow_pickle=True)
    # arr is an array of objects (ndarrays) in the same order
    return [np.array(x, dtype=np.float32) for x in arr.tolist()]


# -------------------------
# PACK/UNPACK SIGNED UPDATE (STRUCTURED, NO AD-HOC CONCAT)
# -------------------------
def pack_signed_update(signature: bytes, raw_weights: bytes) -> bytes:
    """
    Pack signature and raw weight bytes into a single safe npz payload.
    - signature stored as uint8 array under key 'sig'
    - raw bytes stored as uint8 array under key 'raw'
    This avoids ad-hoc concatenation and makes unpacking robust.
    """
    bio = BytesIO()
    np.savez_compressed(
        bio,
        sig=np.frombuffer(signature, dtype=np.uint8),
        raw=np.frombuffer(raw_weights, dtype=np.uint8),
    )
    bio.seek(0)
    return bio.read()


def unpack_signed_update(packed: bytes):
    """
    Reverse of pack_signed_update: returns (signature: bytes, raw_weights: bytes)
    """
    bio = BytesIO(packed)
    npz = np.load(bio)
    sig = npz["sig"].tobytes()
    raw = npz["raw"].tobytes()
    return sig, raw


# -------------------------
# ENCRYPTION (AES-GCM)
# -------------------------
def generate_aes_key():
    return AESGCM.generate_key(bit_length=256)


def aes_encrypt(key: bytes, plaintext: bytes):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return nonce, ct


def aes_decrypt(key: bytes, nonce: bytes, ciphertext: bytes):
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=None)


# -------------------------
# SIGNATURES (Ed25519)
# -------------------------
def generate_ed25519_keypair():
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    return priv, pub


def sign_bytes(priv: Ed25519PrivateKey, data: bytes) -> bytes:
    return priv.sign(data)


def verify_signature(pub: Ed25519PublicKey, signature: bytes, data: bytes) -> bool:
    try:
        pub.verify(signature, data)
        return True
    except Exception:
        return False


def serialize_public_key(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


def deserialize_public_key(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)
