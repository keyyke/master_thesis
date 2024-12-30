import unittest

from src.pre_processing.flags.base_flag import BaseFlag
from src.pre_processing.flags.flag import Flag


class Test_Flags(unittest.TestCase):

    def test_BaseFlag(self):
        flag1 = BaseFlag()
        flag2 = BaseFlag()

        self.assertTrue(flag1 is flag2)
        self.assertTrue(flag2 is flag1)

        self.assertEqual(id(flag2), id(flag1))

    def test_BaseFlag_getters(self):
        flag1 = BaseFlag()
        flag2 = BaseFlag()

        self.assertEqual(flag1.getName(), 'BaseFlag')

        self.assertEqual(flag1.getUUID(), flag2.getUUID())

        self.assertEqual(flag1.getParents(), [BaseFlag()])
        self.assertEqual(flag1.getParents(verbose=True), ['BaseFlag'])

        self.assertEqual(flag1.getAllParents(), [BaseFlag()])
        self.assertEqual(flag1.getAllParents(verbose=True), ['BaseFlag'])

    def test_BaseFlag_contains(self):
        flag1 = BaseFlag()
        flag2 = BaseFlag()

        self.assertTrue(flag1.contains(flag1))
        self.assertTrue(flag1.contains(flag2))

        otherFlag = Flag('Test')
        self.assertFalse(flag1.contains(otherFlag))
        self.assertTrue(otherFlag.contains(flag1))

        otherFlag2 = Flag("Test2", parents=[BaseFlag()])

        self.assertFalse(flag1.contains(otherFlag2))
        self.assertTrue(otherFlag2.contains(flag1))

    def test_Flag(self):
        flag1 = Flag('First')
        flag2 = Flag('Second')

        self.assertEqual(flag1.getName(), 'First')
        self.assertNotEqual(flag1.getUUID(), flag2.getUUID())
        self.assertNotEqual(id(flag1), id(flag2))

    def test_Flag_contains(self):
        flag1 = Flag('First')
        flag2 = Flag('Second')
        flag3 = Flag('Third', parents=[flag1, flag2])
        flag4 = Flag('Fourth', parents=[flag2])
        flag5 = Flag('Fifth', parents=[flag3])

        self.assertTrue(flag3.contains(flag1))
        self.assertTrue(flag3.contains(flag2))
        self.assertTrue(flag3.contains(BaseFlag()))
        self.assertFalse(flag3.contains(flag4))

        self.assertTrue(flag5.contains(flag1))
        self.assertTrue(flag5.contains(flag2))
        self.assertTrue(flag5.contains(flag3))
        self.assertFalse(flag3.contains(flag5))

    def test_Flag_getParents(self):
        flag1 = Flag('First')
        flag2 = Flag('Second')
        flag3 = Flag('Third', parents=[flag1, flag2])
        flag4 = Flag('Fourth', parents=[flag2])
        flag5 = Flag('Fifth', parents=[flag3])
        flag6 = Flag('Sixth', parents=[flag5, flag4])

        self.assertEqual(['Fifth', 'Fourth'], flag6.getParents(verbose=True))
        self.assertEqual([flag5, flag4], flag6.getParents(verbose=False))

        self.assertEqual(['First', 'Second'], flag3.getParents(verbose=True))
        self.assertEqual([flag1, flag2], flag3.getParents(verbose=False))

        self.assertEqual(['BaseFlag'], flag1.getParents(verbose=True))
        self.assertEqual([BaseFlag()], flag1.getParents(verbose=False))

    def test_getAllParents(self):
        flag1 = Flag('First')
        flag2 = Flag('Second')
        flag3 = Flag('Third', parents=[flag1, flag2])
        flag4 = Flag('Fourth', parents=[flag2])
        flag5 = Flag('Fifth', parents=[flag3])
        flag6 = Flag('Sixth', parents=[flag5, flag4])

        self.assertEqual(['Fifth', 'Third', 'First', 'BaseFlag', 'Second', 'Fourth'], flag6.getAllParents(verbose=True))
        self.assertEqual([flag5, flag3, flag1, BaseFlag(), flag2, flag4], flag6.getAllParents(verbose=False))

        self.assertEqual(['BaseFlag'], flag1.getAllParents(verbose=True))
        self.assertEqual([BaseFlag()], flag1.getAllParents(verbose=False))
