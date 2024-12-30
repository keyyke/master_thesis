from src.pre_processing.custom_queues.custom_queue import CustomQueue


class CustomPriorityQueue(CustomQueue):
    def __insert(self, element):
        index = 0

        for i, q in enumerate(super().getAll()):
            if element > q or not (element < q):
                index += 1

        super()._insert(element, index)

    def push(self, element):
        self.__insert(element)


