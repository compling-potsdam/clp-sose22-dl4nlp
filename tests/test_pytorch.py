import unittest

from project.models.fashion import perform_training


class MyTestCase(unittest.TestCase):

    def test_perform_training(self):
        perform_training()


if __name__ == '__main__':
    unittest.main()
