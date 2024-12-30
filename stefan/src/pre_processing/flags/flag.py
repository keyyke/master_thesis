from src.pre_processing.flags.base_flag import BaseFlag
from src.pre_processing.flags.abstract_flag import AbstractFlag


class Flag(AbstractFlag):
    def __init__(self, name: str, parents: list[AbstractFlag] | AbstractFlag = None):
        super().__init__(name)

        if parents is None:
            parents = [BaseFlag()]
        elif isinstance(parents, AbstractFlag):
            parents = [parents]

        self.__parents = parents

    def contains(self, flag: AbstractFlag) -> bool:
        if flag is self or self.getUUID() == flag.getUUID():
            return True

        for parent in self.__parents:
            if parent.contains(flag):
                return True

        return False

    def getParents(self, verbose: bool = False) -> list[str] | list[AbstractFlag]:
        if not verbose:
            return self.__parents

        return [p.getName() for p in self.__parents]

    def getAllParents(self, verbose: bool = False) -> list[str] | list[AbstractFlag]:
        allParents = []

        for parent in self.__parents:
            if not verbose:
                allParents.append(parent)
            else:
                allParents.append(parent.getName())

            allParents.extend(parent.getAllParents(verbose))

        result = []
        for i in allParents:
            if i not in result:
                result.append(i)

        return result
