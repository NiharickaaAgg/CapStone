"""Run every unit test (API, model, logging) with one command: python run_tests.py"""
import sys, unittest
suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
