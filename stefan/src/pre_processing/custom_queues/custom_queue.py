class CustomQueue:
    def __init__(self):
        self.__queue = []
        self.__iter = 0
        self.__reset_iter()

    def _insert(self, element, index):
        assert 0 <= index <= len(self.__queue), 'Index out of range'
        self.__queue.insert(index, element)

    def push(self, element):
        self.__queue.append(element)

    def pushAll(self, elements: list):
        for e in elements:
            self.push(e)

    def pop(self):
        assert not self.empty()

        element = self.__queue[0]
        self.__queue = self.__queue[1:]

        return element

    def popAll(self):
        result = []
        while not self.empty():
            result.append(self.pop())
        return result

    def getAll(self):
        return [i for i in self.__queue]

    def size(self):
        return len(self.__queue)

    def empty(self):
        return len(self.__queue) == 0

    def reset(self):
        self.__queue = []

    def contains(self, element):
        if isinstance(element, str):
            return element in [str(i) for i in self.__queue]
        else:
            return element in self.__queue

    def __len__(self):
        return self.size()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__queue[slice(key.start, key.stop, key.step)]

        elif isinstance(key, int):
            return self.__queue[key]

        elif isinstance(key, str):
            for i in self.__queue:
                if str(i) == str(key):
                    return i

        raise Exception('Invalid Key')

    def __delitem__(self, key):
        if isinstance(key, slice):
            del self.__queue[slice(key.start, key.stop, key.step)]
            return

        elif isinstance(key, int):
            del self.__queue[key]
            return

        elif isinstance(key, str):
            for index, element in enumerate(self.__queue):
                if str(element) == str(key):
                    del self.__queue[index]
                    return

        raise Exception('Invalid Key')

    def __contains__(self, key):
        return self.contains(key)

    def __reset_iter(self):
        self.__iter = 0

    def __iter__(self):
        self.__reset_iter()
        return self

    def __next__(self):
        try:
            element = self.__queue[self.__iter]
            self.__iter += 1
            return element

        except IndexError:
            raise StopIteration

