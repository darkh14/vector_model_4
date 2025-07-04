import json
import logging
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import time
import os

from config import SOURCE_FOLDER, TEMP_FOLDER
from entities import TaskData, StatusResponse, ProcessingTaskResponse

logger = logging.getLogger(__name__)


class TaskStorage:
    """Управление задачами и файлами."""

    def __init__(self):
        self.tasks: Dict[str, TaskData] = {}
        self.lock = asyncio.Lock()

    async def create_task(self, task_id: str) -> TaskData:
        """Создает новую задачу."""
        async with self.lock:
            task = TaskData(task_id=task_id)
            self.tasks[task_id] = task
            logger.info(f"Created task: {task_id}")
            return task

    async def get_task(self, task_id: str) -> Optional[TaskData]:
        """Получает данные задачи."""
        return self.tasks.get(task_id)
    
    async def get_processing_tasks(self, accounting_db: str='') -> List[ProcessingTaskResponse]:
        """Получает данные всех задач."""
        all_tasks = [el for el in self.tasks.values() if el.model_dump().get('status') not in ['READY', 'ERROR']]
        if accounting_db:
            all_tasks = [el for el in all_tasks if el.model_dump().get('accounting_db', '') == accounting_db]

        result = []
        for el in all_tasks:
            result.append(ProcessingTaskResponse(task_id=el.task_id, 
                                                 type=el.type, 
                                                 accounting_db=el.model_dump().get('accounting_db', ''),
                                                 status=el.status))

        return result

    async def update_task(self, task_id: str, **kwargs) -> None:
        """Обновляет данные задачи."""
        async with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Task not found: {task_id}")
                return

            task = self.tasks[task_id]

            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

            # Специальная обработка для времени
            if kwargs.get("status") == "PROCESSING" and task.start_time is None:
                task.start_time = time.time()
            elif kwargs.get("status") == "READY" and task.start_time:
                task.end_time = time.time()

    async def get_task_status(self, task_id: str) -> Optional[StatusResponse]:
        """Получает статус задачи для API."""
        task = await self.get_task(task_id)
        if not task:
            return None

        return StatusResponse(
            status=task.status,
            end_time=task.end_time,
            error=task.error
        )

    async def save_upload_file(self, task_id: str, filename: str, content: bytes) -> Path:
        """Сохраняет загруженный файл."""
        folder = os.path.join(SOURCE_FOLDER, f"{task_id}")
        #os.mkdir(folder) # Пофиксил баг
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{filename}")

        async with asyncio.Lock():
            with open(file_path, "wb") as f:
                f.write(content)

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path # type: ignore

    async def cleanup_task_files(self, task_id: str) -> None:
        """Очищает временные файлы задачи."""
        patterns = [
            SOURCE_FOLDER / f"{task_id}",
            TEMP_FOLDER / f"{task_id}",
        ]

        for pattern in patterns:
            # Для директорий с именем без wildcards
            if pattern.name and '*' not in pattern.name and pattern.exists() and pattern.is_dir():
                try:
                    shutil.rmtree(pattern)
                    logger.info(f"Removed directory: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {pattern}: {e}")
            else:
                # Для паттернов с wildcards
                for file in pattern.parent.glob(pattern.name):
                    try:
                        if file.is_dir():
                            shutil.rmtree(file)
                            logger.info(f"Removed directory: {file}")
                        else:
                            file.unlink()
                            logger.info(f"Removed file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file}: {e}")


# Глобальный экземпляр хранилища
task_storage = TaskStorage()