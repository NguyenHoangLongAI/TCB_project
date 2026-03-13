# Embedding_vectorDB/user_groups_collection.py
"""
User Groups Collection Manager
Quản lý phân quyền theo 4 tầng: group_id > company_id > department_id > user_id
"""

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)
import hashlib
import os
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserGroupsManager:
    """
    Quản lý collection user_groups với 4 tầng phân quyền:
      group_id > company_id > department_id > user_id (PK)

    Mỗi user_id có: cost_llm_tokens, username, password_hash
    Null ở tầng nào = tất cả entity thuộc tầng dưới đều match.
    """

    COLLECTION_NAME = "user_groups"

    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.collection: Optional[Collection] = None
        self._connect()
        self._create_collection()

    # ──────────────────────────────────────────────
    # CONNECTION
    # ──────────────────────────────────────────────

    def _connect(self):
        try:
            try:
                connections.disconnect("default")
            except Exception:
                pass
            connections.connect("default", host=self.host, port=self.port, timeout=10)
            logger.info(f"✅ Connected to Milvus {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Milvus: {e}")
            raise

    # ──────────────────────────────────────────────
    # COLLECTION SCHEMA
    # ──────────────────────────────────────────────

    def _create_collection(self):
        if utility.has_collection(self.COLLECTION_NAME):
            logger.info(f"Collection '{self.COLLECTION_NAME}' already exists – loading.")
            self.collection = Collection(self.COLLECTION_NAME)
            self.collection.load()
            return

        fields = [
            FieldSchema(name="user_id",       dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="group_id",       dtype=DataType.VARCHAR, max_length=100),   # e.g. "Techcombank_group"
            FieldSchema(name="company_id",     dtype=DataType.VARCHAR, max_length=100),   # e.g. "Techcomlife" | ""
            FieldSchema(name="department_id",  dtype=DataType.VARCHAR, max_length=100),   # e.g. "Sell_Techcomlife" | ""
            FieldSchema(name="username",       dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="password_hash",  dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="cost_llm_tokens",dtype=DataType.INT64),                     # cumulative token cost
            # Dummy vector (required by Milvus, use dimension=2 to keep overhead minimal)
            FieldSchema(name="dummy_vector",   dtype=DataType.FLOAT_VECTOR, dim=2),
        ]

        schema = CollectionSchema(fields, description="User groups & access control (4-level hierarchy)")
        self.collection = Collection(self.COLLECTION_NAME, schema=schema, using="default")

        # Index on dummy vector (required to load collection)
        self.collection.create_index(
            field_name="dummy_vector",
            index_params={"metric_type": "L2", "index_type": "FLAT", "params": {}}
        )
        self.collection.load()
        logger.info(f"✅ Collection '{self.COLLECTION_NAME}' created.")

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def _null_str(value: Optional[str]) -> str:
        """Store None/null as empty string (Milvus VARCHAR cannot be NULL)."""
        return value.strip() if value else ""

    # ──────────────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────────────

    def create_user(
        self,
        user_id: str,
        username: str,
        password: str,
        group_id: str,
        company_id: Optional[str] = None,
        department_id: Optional[str] = None,
        initial_tokens: int = 0,
    ) -> bool:
        """Create a new user. Returns False if user_id already exists."""
        try:
            if self._user_exists(user_id):
                logger.warning(f"user_id '{user_id}' already exists.")
                return False

            entities = [
                [user_id],
                [self._null_str(group_id)],
                [self._null_str(company_id)],
                [self._null_str(department_id)],
                [username],
                [self._hash_password(password)],
                [initial_tokens],
                [[0.0, 0.0]],   # dummy vector
            ]
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ User '{user_id}' created.")
            return True
        except Exception as e:
            logger.error(f"❌ create_user error: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one user by user_id."""
        try:
            results = self.collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id","group_id","company_id","department_id",
                                "username","cost_llm_tokens"],
                limit=1,
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"❌ get_user error: {e}")
            return None

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by username + password. Returns user dict or None."""
        try:
            pw_hash = self._hash_password(password)
            results = self.collection.query(
                expr=f'username == "{username}"',
                output_fields=["user_id","group_id","company_id","department_id",
                                "username","password_hash","cost_llm_tokens"],
                limit=10,
            )
            for row in results:
                if row.get("password_hash") == pw_hash:
                    row.pop("password_hash", None)
                    return row
            return None
        except Exception as e:
            logger.error(f"❌ authenticate error: {e}")
            return None

    def update_token_cost(self, user_id: str, tokens_used: int) -> bool:
        """Increment cost_llm_tokens for a user atomically (read-modify-write)."""
        try:
            row = self.get_user(user_id)
            if not row:
                logger.warning(f"user_id '{user_id}' not found for token update.")
                return False

            current = row.get("cost_llm_tokens", 0)
            new_total = current + tokens_used

            # Milvus doesn't support in-place update → delete + re-insert
            # Fetch full row first (need password_hash)
            full = self.collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id","group_id","company_id","department_id",
                                "username","password_hash","cost_llm_tokens"],
                limit=1,
            )
            if not full:
                return False
            r = full[0]

            self.collection.delete(f'user_id in ["{user_id}"]')

            entities = [
                [r["user_id"]],
                [r["group_id"]],
                [r["company_id"]],
                [r["department_id"]],
                [r["username"]],
                [r["password_hash"]],
                [new_total],
                [[0.0, 0.0]],
            ]
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ user '{user_id}' tokens: {current} → {new_total}")
            return True
        except Exception as e:
            logger.error(f"❌ update_token_cost error: {e}")
            return False

    def list_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all users (without password_hash)."""
        try:
            results = self.collection.query(
                expr='user_id != ""',
                output_fields=["user_id","group_id","company_id","department_id",
                                "username","cost_llm_tokens"],
                limit=limit,
            )
            return results
        except Exception as e:
            logger.error(f"❌ list_users error: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        try:
            self.collection.delete(f'user_id in ["{user_id}"]')
            self.collection.flush()
            logger.info(f"✅ User '{user_id}' deleted.")
            return True
        except Exception as e:
            logger.error(f"❌ delete_user error: {e}")
            return False

    # ──────────────────────────────────────────────
    # PERMISSION HELPERS
    # ──────────────────────────────────────────────

    def get_user_permissions(self, user_id: str) -> Optional[Dict[str, str]]:
        """
        Return permission scope for a user:
          { group_id, company_id, department_id, user_id }
        Empty string = wildcard (all children match).
        """
        row = self.get_user(user_id)
        if not row:
            return None
        return {
            "group_id":      row.get("group_id", ""),
            "company_id":    row.get("company_id", ""),
            "department_id": row.get("department_id", ""),
            "user_id":       user_id,
        }

    # ──────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────

    def _user_exists(self, user_id: str) -> bool:
        try:
            r = self.collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["user_id"],
                limit=1,
            )
            return bool(r)
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# SEED FUNCTION  –  5 sample users
# ─────────────────────────────────────────────────────────────────────────────

def seed_sample_users(manager: UserGroupsManager):
    """
    Insert 5 sample users covering different permission levels.

    Example 1 (all Sell dept of Techcomlife):
        group_id=Techcombank_group, company_id=Techcomlife,
        department_id=Sell_Techcomlife, user_id=user_sell_001

    Example 2 (all Techcombank employees):
        group_id=Techcombank_group, company_id="", department_id="", user_id=…
    """
    users = [
        {
            "user_id":       "admin_001",
            "username":      "admin",
            "password":      "admin@123",
            "group_id":      "Techcombank_group",
            "company_id":    "",                    # wildcard → all companies
            "department_id": "",                    # wildcard → all departments
        },
        {
            "user_id":       "tcl_manager_001",
            "username":      "tcl_manager",
            "password":      "tcl@123",
            "group_id":      "Techcombank_group",
            "company_id":    "Techcomlife",
            "department_id": "",                    # wildcard → all depts in Techcomlife
        },
        {
            "user_id":       "sell_001",
            "username":      "sell_user1",
            "password":      "sell@123",
            "group_id":      "Techcombank_group",
            "company_id":    "Techcomlife",
            "department_id": "Sell_Techcomlife",
        },
        {
            "user_id":       "sell_002",
            "username":      "sell_user2",
            "password":      "sell@456",
            "group_id":      "Techcombank_group",
            "company_id":    "Techcomlife",
            "department_id": "Sell_Techcomlife",
        },
        {
            "user_id":       "hr_001",
            "username":      "hr_user1",
            "password":      "hr@123",
            "group_id":      "Techcombank_group",
            "company_id":    "Techcomlife",
            "department_id": "HR_Techcomlife",
        },
    ]

    for u in users:
        ok = manager.create_user(
            user_id=u["user_id"],
            username=u["username"],
            password=u["password"],
            group_id=u["group_id"],
            company_id=u.get("company_id"),
            department_id=u.get("department_id"),
            initial_tokens=0,
        )
        status = "✅ created" if ok else "⚠️  already exists"
        logger.info(f"  {u['user_id']} ({u['username']}): {status}")


if __name__ == "__main__":
    import os
    mgr = UserGroupsManager(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
    )
    logger.info("Seeding sample users …")
    seed_sample_users(mgr)
    logger.info("Done. Current users:")
    for u in mgr.list_users():
        logger.info(f"  {u}")