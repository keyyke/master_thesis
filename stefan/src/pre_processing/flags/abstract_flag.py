from __future__ import annotations

import uuid
from abc import ABC, abstractmethod


class AbstractFlag(ABC):
    def __init__(self, name: str):
        self.__name = name
        self.__uuid = uuid.uuid4()

    def getName(self) -> str:
        return self.__name

    def getUUID(self) -> uuid:
        return self.__uuid

    @abstractmethod
    def contains(self, flag: __init__) -> bool:
        pass

    @abstractmethod
    def getParents(self, verbose: bool = False) -> list[str] | list[AbstractFlag]:
        pass

    @abstractmethod
    def getAllParents(self, verbose: bool = False) -> list[str] | list[AbstractFlag]:
        pass
