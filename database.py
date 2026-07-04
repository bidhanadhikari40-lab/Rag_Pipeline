import hashlib
import hmac
import os
import sqlite3
from pathlib import Path


DB_PATH = Path("data/chatbot.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact_detail TEXT,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                address TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scraped_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                date TEXT,
                url TEXT NOT NULL UNIQUE,
                scraped_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES admin_info(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """
        )


def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    ).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, digest = stored_hash.split("$", 2)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    ).hex()
    return hmac.compare_digest(candidate, digest)


def create_admin(
    name: str,
    contact_detail: str,
    email: str,
    password: str,
    address: str,
) -> tuple[bool, str]:
    if not name.strip() or not email.strip() or not password:
        return False, "Name, email, and password are required."

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO admin_info
                    (name, contact_detail, email, password, address)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    contact_detail.strip(),
                    email.strip().lower(),
                    hash_password(password),
                    address.strip(),
                ),
            )
    except sqlite3.IntegrityError:
        return False, "That email is already registered."

    return True, "Account created. You can sign in now."


def authenticate(email: str, password: str) -> sqlite3.Row | None:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT * FROM admin_info WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()

    if user and verify_password(password, user["password"]):
        return user
    return None


def upsert_scraped_pages(pages: list[dict]) -> int:
    with get_connection() as conn:
        for page in pages:
            conn.execute(
                """
                INSERT INTO scraped_pages (title, content, date, url)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    date = excluded.date,
                    scraped_at = CURRENT_TIMESTAMP
                """,
                (
                    page.get("title", "Untitled"),
                    page.get("content", ""),
                    page.get("date", ""),
                    page.get("url", ""),
                ),
            )
    return len(pages)


def load_pages_from_db() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT title, content, date, url, scraped_at
            FROM scraped_pages
            ORDER BY scraped_at DESC, title ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def create_chat_session(admin_id: int, title: str = "New chat") -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO chat_sessions (admin_id, title) VALUES (?, ?)",
            (admin_id, title.strip() or "New chat"),
        )
        return int(cursor.lastrowid)


def list_chat_sessions(admin_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.admin_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC, s.id DESC
            """,
            (admin_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat_session(admin_id: int, session_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chat_sessions
            WHERE admin_id = ? AND id = ?
            """,
            (admin_id, session_id),
        ).fetchone()
    return dict(row) if row else None


def rename_chat_session(admin_id: int, session_id: int, title: str) -> None:
    cleaned_title = title.strip()[:80] or "New chat"
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE chat_sessions
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE admin_id = ? AND id = ?
            """,
            (cleaned_title, admin_id, session_id),
        )


def delete_chat_session(admin_id: int, session_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM chat_sessions WHERE admin_id = ? AND id = ?",
            (admin_id, session_id),
        )


def load_chat_messages(admin_id: int, session_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT m.role, m.content, m.created_at
            FROM chat_messages m
            INNER JOIN chat_sessions s ON s.id = m.session_id
            WHERE s.admin_id = ? AND m.session_id = ?
            ORDER BY m.id ASC
            """,
            (admin_id, session_id),
        ).fetchall()
    return [dict(row) for row in rows]


def add_chat_message(admin_id: int, session_id: int, role: str, content: str) -> None:
    if role not in {"user", "assistant"} or not content:
        return

    with get_connection() as conn:
        session = conn.execute(
            "SELECT id, title FROM chat_sessions WHERE admin_id = ? AND id = ?",
            (admin_id, session_id),
        ).fetchone()
        if not session:
            return

        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )

        if role == "user" and session["title"] == "New chat":
            generated_title = " ".join(content.strip().split())[:60] or "New chat"
            conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ?",
                (generated_title, session_id),
            )
