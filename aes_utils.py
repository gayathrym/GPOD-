from cryptography.fernet import Fernet

# Master key used to encrypt user SKUIDs (store this securely!)
MASTER_KEY = b"nRU4WV7qt59kS0mkIPFvtj4FyFAMaVo8p-DUXlxzFW8="
master_cipher = Fernet(MASTER_KEY)

def generate_skuid() -> bytes:
    """Generate a new user-specific encryption key (SKUID)."""
    return Fernet.generate_key()

def encrypt_skuid(skuid: bytes) -> str:
    """Encrypt the SKUID using a master key."""
    return master_cipher.encrypt(skuid).decode()

def decrypt_skuid(encrypted_skuid: str) -> bytes:
    """Decrypt the SKUID using the master key."""
    return master_cipher.decrypt(encrypted_skuid.encode())

def encrypt_with_key(text: str, key: bytes) -> str:
    """Encrypt data using a user-specific key."""
    cipher = Fernet(key)
    return cipher.encrypt(text.encode()).decode()

def decrypt_with_key(token: str, key: bytes) -> str:
    """Decrypt data using a user-specific key."""
    cipher = Fernet(key)
    return cipher.decrypt(token.encode()).decode()
