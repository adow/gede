# coding=utf-8
#
#

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64


def encrypt_aes(plaintext: str, password: str) -> str:
    """
    Encrypt a string using AES encryption with a password.

    Args:
        plaintext: The string to encrypt
        password: The password used for encryption

    Returns:
        A string containing the base64-encoded salt, IV, and encrypted data
    """
    # Generate a random salt for the key derivation
    salt = os.urandom(16)

    # Derive a key from the password
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode("utf-8"))

    # Generate a random IV
    iv = os.urandom(16)

    # Convert plaintext to bytes
    plaintext_bytes = plaintext.encode("utf-8")

    # Pad the plaintext
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext_bytes) + padder.finalize()

    # Create an encryptor
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Encrypt the data
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    # Combine salt, IV, and ciphertext and encode as base64
    encrypted_data = base64.b64encode(salt + iv + ciphertext).decode("utf-8")

    return encrypted_data


def decrypt_aes(encrypted_data: str, password: str) -> str:
    """
    Decrypt an AES-encrypted string using a password.

    Args:
        encrypted_data: The base64-encoded string containing salt, IV, and encrypted data
        password: The password used for decryption

    Returns:
        The decrypted plaintext string
    """
    # Decode the base64 data
    raw_data = base64.b64decode(encrypted_data)

    # Extract salt (first 16 bytes), IV (next 16 bytes), and ciphertext
    salt = raw_data[:16]
    iv = raw_data[16:32]
    ciphertext = raw_data[32:]

    # Derive the key from the password and salt
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode("utf-8"))

    # Create a decryptor
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the ciphertext
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the plaintext
    unpadder = padding.PKCS7(128).unpadder()
    plaintext_bytes = unpadder.update(padded_plaintext) + unpadder.finalize()

    # Convert bytes to string
    plaintext = plaintext_bytes.decode("utf-8")

    return plaintext


# Example usage
def test_aes_encryption():
    original_text = "This is a secret message"
    password = "my-secure-password"

    # Encrypt
    encrypted = encrypt_aes(original_text, password)
    print(f"Encrypted: {encrypted}")

    # Decrypt
    decrypted = decrypt_aes(encrypted, password)
    print(f"Decrypted: {decrypted}")

    # Verify
    assert original_text == decrypted
    print("Encryption and decryption successful!")


if __name__ == "__main__":
    test_aes_encryption()
