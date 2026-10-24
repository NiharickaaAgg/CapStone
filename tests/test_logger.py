import os, unittest
from cslib import logger, config

class LoggerTest(unittest.TestCase):
    def test_train_log_written_to_test_dir(self):
        logger.update_train_log("all", "ridge", 1.0, 2.0, 0.1, "t", test=True)
        files = [f for f in os.listdir(config.log_dir(True)) if f.startswith("train")]
        self.assertTrue(files)

    def test_predict_log_appends(self):
        logger.update_predict_log("all", "2019-01-01", 10.0, 0.1, "t", test=True)
        p = [f for f in os.listdir(config.log_dir(True)) if f.startswith("predict")][0]
        n1 = len(open(os.path.join(config.log_dir(True), p)).readlines())
        logger.update_predict_log("all", "2019-01-02", 11.0, 0.1, "t", test=True)
        n2 = len(open(os.path.join(config.log_dir(True), p)).readlines())
        self.assertEqual(n2, n1 + 1)

    def test_prod_dir_untouched(self):
        self.assertNotEqual(config.log_dir(True), config.log_dir(False))
