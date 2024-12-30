import unittest

from src.pre_processing.custom_queues.custom_priority_queue import CustomPriorityQueue
from src.pre_processing.custom_queues.custom_queue import CustomQueue
from src.pre_processing.tasks.task import Task


class Test_simple_queue(unittest.TestCase):
    def setUp(self):
        self.task1 = Task('TestTask1')
        self.task2 = Task('TestTask2')
        self.task3 = Task('TestTask3')
        self.task4 = Task('TestTask4')
        self.task5 = Task('TestTask5')

    def test_push_and_pop_simple(self):
        queue = CustomQueue()

        queue.push(self.task1)
        queue.push(self.task2)
        queue.push(self.task3)

        self.assertEqual(queue.size(), 3)

        self.assertEqual([self.task1, self.task2, self.task3], queue.getAll())

        self.assertTrue(self.task1 is queue.pop())

        queue.push(self.task4)

        self.assertTrue(self.task2 is queue.pop())
        self.assertEqual([self.task3, self.task4], queue.getAll())
        self.assertTrue(self.task3 is queue.pop())

        self.assertEqual(queue.size(), 1)

        self.assertTrue(self.task4 is queue.pop())

        self.assertTrue(queue.empty())

    def test_push_and_pop_all(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2, self.task3, self.task4]

        queue.pushAll(tasks)
        self.assertEqual(queue.size(), 4)

        self.assertEqual(tasks, queue.popAll())
        self.assertTrue(queue.empty())

    def test_reset(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2, self.task3, self.task4]
        queue.pushAll(tasks)

        self.assertEqual(queue.size(), 4)
        queue.reset()
        self.assertTrue(queue.empty())

    def test_contains(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2, self.task3]
        queue.pushAll(tasks)

        self.assertTrue(queue.contains(self.task1))
        self.assertTrue(queue.contains(self.task2))
        self.assertFalse(queue.contains(self.task4))

        self.assertTrue(queue.contains('TestTask1'))
        self.assertTrue(queue.contains('TestTask2'))
        self.assertFalse(queue.contains('TestTask4'))

        task = queue.pop()

        self.assertFalse(queue.contains(self.task1))
        self.assertFalse(queue.contains('TestTask1'))

    def test_contains_in(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2, self.task3]
        queue.pushAll(tasks)

        self.assertTrue(self.task1 in queue)
        self.assertFalse(self.task4 in queue)

        self.assertTrue('TestTask1' in queue)
        self.assertFalse('TestTask4' in queue)

        task = queue.pop()

        self.assertFalse(self.task1 in queue)
        self.assertFalse('TestTask1' in queue)

    def test_length(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2]
        queue.pushAll(tasks)

        self.assertEqual(len(queue), 2)

    def test_getitem(self):
        queue = CustomQueue()

        newTask = Task("TestTask1")

        tasks = [self.task1, self.task2, self.task3, self.task4, self.task5, newTask]
        queue.pushAll(tasks)

        self.assertTrue(self.task2 is queue[1])
        self.assertTrue(self.task5 is queue[-2])
        self.assertTrue(Task, type(queue[3]))

        self.assertEqual(tasks[1:3], queue[1:3])
        self.assertEqual(tasks[-1:-4:2], queue[-1:-4:2])
        self.assertEqual(tasks[1:100:3], queue[1:100:3])

        self.assertEqual(type(queue[1:3:5]), list)

        self.assertTrue(self.task1 is queue['TestTask1'])
        self.assertFalse(newTask is queue['TestTask1'])

        with self.assertRaises(Exception):
            element = queue['TestTask7']
        with self.assertRaises(Exception):
            element = queue[99]
        with self.assertRaises(Exception):
            element = queue[4.3]

    def test_iter(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2, self.task3, self.task4, self.task5]
        queue.pushAll(tasks)
        self.assertEqual(5, len(queue))

        for i, e in enumerate(queue):
            if i > 2:
                break
            self.assertEqual(str(e), 'TestTask' + str(i + 1))

        self.assertEqual(5, len(queue))

        counter = 0
        for i, e in enumerate(queue):
            counter += 1
            self.assertEqual(str(e), 'TestTask' + str(i + 1))

        self.assertEqual(counter, len(queue))

    def test_insert(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2]
        queue.pushAll(tasks)

        queue._insert(self.task3, 0)
        queue._insert(self.task4, 3)
        queue._insert(self.task5, 2)

        self.assertEqual([self.task3, self.task1, self.task5, self.task2, self.task4], queue.getAll())

        with self.assertRaises(AssertionError):
            queue._insert(self.task4, 6)

        with self.assertRaises(AssertionError):
            queue._insert(self.task4, -2)

    def test_delete_int(self):
        queue = CustomQueue()

        newTask = Task("TestTask1")

        tasks = [self.task1, self.task2, self.task3, self.task4, self.task5, newTask]
        queue.pushAll(tasks)

        del queue[3]
        self.assertEqual([self.task1, self.task2, self.task3, self.task5, newTask], queue.getAll())

    def test_delete_slice(self):
        queue = CustomQueue()

        newTask = Task("TestTask1")

        tasks = [self.task1, self.task2, self.task3, self.task4, self.task5, newTask]
        queue.pushAll(tasks)

        del queue[1:-2: 2]

        self.assertEqual([self.task1, self.task3, self.task5, newTask], queue.getAll())

    def test_delete_string(self):
        queue = CustomQueue()

        newTask = Task("TestTask1")

        tasks = [self.task1, self.task2, self.task3, self.task4, self.task5, newTask]
        queue.pushAll(tasks)

        del queue["TestTask1"]

        self.assertEqual(tasks[1:], queue.getAll())

    def test_delete_exception(self):
        queue = CustomQueue()

        tasks = [self.task1, self.task2, self.task3, self.task4, self.task5]
        queue.pushAll(tasks)

        with self.assertRaises(Exception):
            del queue['TestTask7']
        with self.assertRaises(Exception):
            del queue[99]
        with self.assertRaises(Exception):
            del queue[4.3]


class Test_priority_queue(unittest.TestCase):
    def setUp(self):
        self.task1 = Task('TestTask1', priority=1)
        self.task2 = Task('TestTask2', priority=2)
        self.task3 = Task('TestTask3', priority=3)
        self.task4 = Task('TestTask4', priority=3)
        self.task5 = Task('TestTask5', priority=None)
        self.task6 = Task('TestTask6', priority=None)

    def test_ordering(self):
        queue = CustomPriorityQueue()

        queue.push(self.task2)
        queue.push(self.task5)
        queue.push(self.task3)

        queue.pushAll([self.task4, self.task6, self.task1])

        self.assertEqual(6, queue.size())

        orderedTasks = [self.task1, self.task2, self.task3, self.task4, self.task5, self.task6]

        self.assertEqual(orderedTasks, queue.getAll())
        self.assertEqual(orderedTasks, queue.popAll())

        self.assertTrue(queue.empty())
