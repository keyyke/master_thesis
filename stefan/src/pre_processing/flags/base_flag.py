from src.pre_processing.flags.abstract_flag import AbstractFlag
class BaseFlag(AbstractFlag):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(BaseFlag, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        super(BaseFlag, self).__init__('BaseFlag')

    def contains(self, flag: AbstractFlag) -> bool:
        if flag is self or self.getUUID() == flag.getUUID():
            return True

        return False

    def getParents(self, verbose: bool = False) -> list[str] | list[AbstractFlag]:
        if not verbose:
            return [self]
        else:
            return [self.getName()]

    def getAllParents(self, verbose: bool = False) -> list[str] | list[AbstractFlag]:
        return self.getParents(verbose)
