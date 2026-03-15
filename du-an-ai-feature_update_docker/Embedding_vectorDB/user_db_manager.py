# Embedding_vectorDB/user_db_manager.py  (FIXED)
"""
FIXES:
  Bug 2 — update_token_cost(): thêm flush() + release()/load() sau insert
           để đảm bảo ghi xuống storage và đọc lại được ngay lập tức
  Bug alias — dùng _CONN_ALIAS = "user_db_conn" nhất quán xuyên suốt,
              tránh lẫn với alias "default" của MilvusManager
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, db, utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_DATABASE_NAME = "user_db"
_CONN_ALIAS = "user_db_conn"   # alias riêng, không đụng alias "default"


class UserDBManager:

    COLLECTION_NAME = "user_groups"

    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.collection: Optional[Collection] = None
        self._connect_and_init()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect_and_init(self):
        try:
            try:
                connections.disconnect(_CONN_ALIAS)
            except Exception:
                pass
            connections.connect(alias=_CONN_ALIAS, host=self.host, port=self.port, timeout=10)
            logger.info(f"✅ UserDBManager [{_CONN_ALIAS}] {self.host}:{self.port}")

            self._ensure_database_exists()
            db.using_database(USER_DATABASE_NAME, using=_CONN_ALIAS)
            logger.info(f"✅ Switched to database: {USER_DATABASE_NAME}")

            self._create_or_load_collection()
        except Exception as e:
            logger.error(f"❌ UserDBManager init failed: {e}")
            raise

    def _ensure_database_exists(self):
        try:
            existing = db.list_database(using=_CONN_ALIAS)
            if USER_DATABASE_NAME not in existing:
                db.create_database(USER_DATABASE_NAME, using=_CONN_ALIAS)
                logger.info(f"✅ Created DB: {USER_DATABASE_NAME}")
        except Exception as e:
            logger.warning(f"⚠️ Multi-db may not be supported, fallback to default: {e}")

    def _create_or_load_collection(self):
        if utility.has_collection(self.COLLECTION_NAME, using=_CONN_ALIAS):
            self.collection = Collection(self.COLLECTION_NAME, using=_CONN_ALIAS)
            self.collection.load()
            logger.info(f"✅ Loaded collection '{self.COLLECTION_NAME}'")
            return

        fields = [
            FieldSchema(name="user_id",        dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="group_id",        dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="company_id",      dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="department_id",   dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="username",        dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="password_hash",   dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="cost_llm_tokens", dtype=DataType.INT64),
            FieldSchema(name="dummy_vector",    dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields, description=f"User groups. DB: {USER_DATABASE_NAME}")
        self.collection = Collection(self.COLLECTION_NAME, schema=schema, using=_CONN_ALIAS)
        self.collection.create_index(
            field_name="dummy_vector",
            index_params={"metric_type": "L2", "index_type": "FLAT", "params": {}},
        )
        self.collection.load()
        logger.info(f"✅ Created collection '{self.COLLECTION_NAME}' in '{USER_DATABASE_NAME}'")

    def _reconnect_if_needed(self):
        try:
            _ = self.collection.num_entities
        except Exception:
            logger.warning("UserDBManager: stale connection, reconnecting…")
            self._connect_and_init()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_password(p: str) -> str:
        return hashlib.sha256(p.encode()).hexdigest()

    @staticmethod
    def _s(v: Optional[str]) -> str:
        return v.strip() if v else ""

    def _user_exists(self, user_id: str) -> bool:
        try:
            return bool(self.collection.query(
                expr=f'user_id == "{user_id}"', output_fields=["user_id"], limit=1
            ))
        except Exception:
            return False

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create_user(
        self,
        user_id: str, username: str, password: str, group_id: str,
        company_id: Optional[str] = None, department_id: Optional[str] = None,
        initial_tokens: int = 0,
    ) -> bool:
        self._reconnect_if_needed()
        try:
            if self._user_exists(user_id):
                logger.warning(f"user_id '{user_id}' already exists.")
                return False
            self.collection.insert([
                [user_id], [self._s(group_id)], [self._s(company_id)],
                [self._s(department_id)], [username],
                [self._hash_password(password)], [initial_tokens], [[0.0, 0.0]],
            ])
            self.collection.flush()
            logger.info(f"✅ User '{user_id}' created (tokens={initial_tokens})")
            return True
        except Exception as e:
            logger.error(f"❌ create_user error: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        self._reconnect_if_needed()
        try:
            rows = self.collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id", "group_id", "company_id",
                               "department_id", "username", "cost_llm_tokens"],
                limit=1,
                consistency_level="Strong",
            )
            return rows[0] if rows else None
        except Exception as e:
            logger.error(f"❌ get_user error: {e}")
            return None

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        self._reconnect_if_needed()
        try:
            pw_hash = self._hash_password(password)
            rows = self.collection.query(
                expr=f'username == "{username}"',
                output_fields=["user_id","group_id","company_id","department_id",
                               "username","password_hash","cost_llm_tokens"],
                limit=10,
            )
            for row in rows:
                if row.get("password_hash") == pw_hash:
                    row.pop("password_hash", None)
                    return row
            return None
        except Exception as e:
            logger.error(f"❌ authenticate error: {e}")
            return None

    def update_token_cost(self, user_id: str, tokens_used: int) -> bool:
        """
        Cộng thêm tokens_used vào cost_llm_tokens.

        Milvus không support UPDATE — phải delete + re-insert.
        Dùng consistency_level="Strong" khi query thay vì release()/load()
        vì release/load tốn 3-8s trên Milvus v2.3 → gây timeout.
        """
        self._reconnect_if_needed()
        try:
            # Đọc row hiện tại với consistency Strong để lấy đúng giá trị mới nhất
            full = self.collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id", "group_id", "company_id", "department_id",
                               "username", "password_hash", "cost_llm_tokens"],
                limit=1,
                consistency_level="Strong",
            )
            if not full:
                logger.warning(f"user_id '{user_id}' not found.")
                return False

            r         = full[0]
            old_total = r.get("cost_llm_tokens") or 0
            new_total = old_total + tokens_used

            # Delete row cũ
            self.collection.delete(f'user_id in ["{user_id}"]')

            # Insert row mới với token đã cộng
            self.collection.insert([
                [r["user_id"]],
                [r.get("group_id",      "")],
                [r.get("company_id",    "")],
                [r.get("department_id", "")],
                [r.get("username",      "")],
                [r.get("password_hash", "")],
                [new_total],
                [[0.0, 0.0]],
            ])

            # flush() để commit xuống storage — không cần release/load
            self.collection.flush()

            logger.info(f"✅ Token updated: {user_id} {old_total} → {new_total} (+{tokens_used})")
            return True

        except Exception as e:
            logger.error(f"❌ update_token_cost error: {e}")
            return False

    def list_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        self._reconnect_if_needed()
        try:
            return self.collection.query(
                expr='user_id != ""',
                output_fields=["user_id","group_id","company_id","department_id","username","cost_llm_tokens"],
                limit=limit,
            )
        except Exception as e:
            logger.error(f"❌ list_users error: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        self._reconnect_if_needed()
        try:
            self.collection.delete(f'user_id in ["{user_id}"]')
            self.collection.flush()
            logger.info(f"✅ User '{user_id}' deleted.")
            return True
        except Exception as e:
            logger.error(f"❌ delete_user error: {e}")
            return False

    # ── Permissions ───────────────────────────────────────────────────────────

    def get_user_permissions(self, user_id: str) -> Optional[Dict[str, str]]:
        row = self.get_user(user_id)
        if not row:
            return None
        return {
            "group_id":      row.get("group_id",      ""),
            "company_id":    row.get("company_id",    ""),
            "department_id": row.get("department_id", ""),
            "user_id":       user_id,
        }

    def build_acl_expression(self, user_id: str) -> str:
        perms = self.get_user_permissions(user_id)
        if not perms:
            return ""
        field_map = {
            "group_id":      "acl_group_id",
            "company_id":    "acl_company_id",
            "department_id": "acl_department_id",
            "user_id":       "acl_user_id",
        }
        clauses = []
        for pk, mf in field_map.items():
            val = (perms.get(pk) or "").strip()
            if val:
                clauses.append(f'({mf} == "" or {mf} == "{val}")')
        return " and ".join(clauses)

    # ── Health ────────────────────────────────────────────────────────────────

    def get_database_info(self) -> Dict[str, Any]:
        self._reconnect_if_needed()
        try:
            return {
                "database": USER_DATABASE_NAME, "collection": self.COLLECTION_NAME,
                "user_count": self.collection.num_entities,
                "host": self.host, "port": self.port,
            }
        except Exception as e:
            return {"error": str(e)}

    def health_check(self) -> bool:
        try:
            self._reconnect_if_needed()
            _ = self.collection.num_entities
            return True
        except Exception:
            return False


# ── Seed ──────────────────────────────────────────────────────────────────────

def seed_sample_users(manager: UserDBManager):
    users = [
        {"user_id":"admin_001",      "username":"admin",       "password":"admin@123",
         "group_id":"Techcombank_group","company_id":"",           "department_id":""},
        {"user_id":"tcl_manager_001","username":"tcl_manager", "password":"tcl@123",
         "group_id":"Techcombank_group","company_id":"Techcomlife","department_id":""},
        {"user_id":"sell_001",       "username":"sell_user1",  "password":"sell@123",
         "group_id":"Techcombank_group","company_id":"Techcomlife","department_id":"Sell_Techcomlife"},
        {"user_id":"sell_002",       "username":"sell_user2",  "password":"sell@456",
         "group_id":"Techcombank_group","company_id":"Techcomlife","department_id":"Sell_Techcomlife"},
        {"user_id":"hr_001",         "username":"hr_user1",    "password":"hr@123",
         "group_id":"Techcombank_group","company_id":"Techcomlife","department_id":"HR_Techcomlife"},
    ]
    for u in users:
        ok = manager.create_user(**u, initial_tokens=0)
        logger.info(f"  {u['user_id']}: {'✅ created' if ok else '⚠️ already exists'}")


# ── Singleton ─────────────────────────────────────────────────────────────────

_user_db_manager: Optional[UserDBManager] = None


def get_user_db_manager(host: str = None, port: str = None) -> UserDBManager:
    global _user_db_manager
    if _user_db_manager is None:
        h = host or os.getenv("MILVUS_HOST", "localhost")
        p = port or os.getenv("MILVUS_PORT", "19530")
        _user_db_manager = UserDBManager(host=h, port=p)
    return _user_db_manager


user_db_manager = get_user_db_manager


if __name__ == "__main__":
    mgr = UserDBManager(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    print(mgr.get_database_info())
    seed_sample_users(mgr)
    for u in mgr.list_users():
        print(u)