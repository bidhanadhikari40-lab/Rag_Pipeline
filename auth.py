"""
Simple local authentication using a JSON file.
Passwords are stored as salted SHA-256 hashes (never plain text).
"""

import json
import os
import hashlib
import hmac
import secrets

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register_user(username: str, password: str, email: str = "") -> tuple[bool, str]:
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    users = _load_users()
    if username in users:
        return False, "Username already exists. Please choose another."

    salt = secrets.token_hex(16)
    users[username] = {
        "email": email.strip(),
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return True, "Account created successfully. You can now log in."


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    users = _load_users()

    if username not in users:
        return False, "Invalid username or password."

    record = users[username]
    expected_hash = record["password_hash"]
    actual_hash = _hash_password(password, record["salt"])

    if hmac.compare_digest(expected_hash, actual_hash):
        return True, "Login successful."
    return False, "Invalid username or password."


def user_exists(username: str) -> bool:
    users = _load_users()
    return username.strip().lower() in users
