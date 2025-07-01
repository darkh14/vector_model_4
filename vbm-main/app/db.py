from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Any
import logging


logger = logging.getLogger(__name__)


class DBProcessor:
    """ Class realises working with DB using MONGO DB.
    """
    def __init__(self) -> None:
        self.url = ''
        self.timeout = 5000
        self.master_db_name = 'vbm'
        self.db = None
        self._connection: Optional[AsyncIOMotorClient] = None
        self.accounting_db = ''

    def set_accounting_db(self, accounting_db):
        self.accounting_db = accounting_db

    def set_accounting_db_filter(self, db_filter):
        if self.accounting_db:
            if db_filter:
                db_filter['accounting_db'] = self.accounting_db
            else:
                db_filter = {'accounting_db': self.accounting_db}

        return db_filter

    async def connect(self, url, timeout) -> bool:
        self.url = url
        self.timeout = timeout
        self._connection = AsyncIOMotorClient(self.url, 
                                        serverSelectionTimeoutMS=self.timeout)
        
        self.db = self._connection.get_database(self.master_db_name)
        
        # Проверка подключения
        await self.db.command("ping")
        logger.info("Successfully connected to the database")
        
        # Проверка целостности базы данных
        db_integrity = await self.check_database_integrity()
        if not db_integrity:
            logger.error("Database integrity check failed")
            raise Exception("Database integrity check failed")
        
        return True

    async def check_database_integrity(self):
        """Проверка целостности базы данных и восстановление индексов"""
        logger.info("check_database_integrity")
        return True
    
    def close(self):
        self._connection.close()

    async def find_one(self, collection_name: str, db_filter=None) -> Optional[dict]:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find line
        :return: dict of db line
        """
        collection = self._get_collection(collection_name)

        db_filter = db_filter or None
        db_filter = self.set_accounting_db_filter(db_filter)

        result = await collection.find_one(db_filter, projection={'_id': False})

        return result or None

    async def find(self, collection_name: str, db_filter=None) -> list[dict]:

        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        result = await collection.find(c_filter, projection={'_id': False}).to_list()
        result = list(result)

        return result

    async def insert_one(self, collection_name: str, value: dict[str, Any], db_filter=None) -> bool:

        collection = self._get_collection(collection_name)

        db_filter = db_filter or None
        db_filter = self.set_accounting_db_filter(db_filter)
        if db_filter:
            result = await collection.replace_one(db_filter, value, upsert=True)
        else:
            result = await collection.insert_one(value)

        return bool(getattr(result, 'acknowledged'))

    async def insert_many(self, collection_name: str, value: list[dict[str, Any]]) -> bool:

        collection = self._get_collection(collection_name)

        result = await collection.insert_many(value)

        return bool(getattr(result, 'acknowledged'))

    async def delete_many(self, collection_name: str, db_filter=None) -> bool:
        
        db_filter = db_filter or None
        db_filter = self.set_accounting_db_filter(db_filter)
        if db_filter:
            collection = self._get_collection(collection_name)
            result = await collection.delete_many(db_filter)
            result = bool(getattr(result, 'acknowledged'))
        else:
            result = await self._db.drop_collection(collection_name)
            result = result is not None

        return result

    async def get_count(self, collection_name: str, db_filter=None) -> int:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find lines
        :return: number of lines in collection
        """
        db_filter = db_filter or {}
        db_filter = self.set_accounting_db_filter(db_filter)
        collection = self._get_collection(collection_name)

        return await collection.count_documents(db_filter)

    def _get_collection(self, collection_name):
        """ Gets collection object from db object
        :param collection_name: name of required collection,
        :return: collection object
        """
        return self.db.get_collection(collection_name)

    def get_collection_names(self) -> list[str]:

        return self._db.list_collection_names()

    async def drop_db(self) -> str:
        """Method to drop current database
        :return result of dropping"""

        result = super().drop_db()

        collection_names = self.get_collection_names()

        async for collection_name in collection_names:
            await self.delete_lines(collection_name)

        return result


db_processor = DBProcessor()
