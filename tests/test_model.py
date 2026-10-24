import shutil, tempfile, unittest
from cslib import model, ingest
from tests.helpers import make_synthetic_data

class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        make_synthetic_data(cls.d)
        cls.out = model.train(data_dir=cls.d, test=True, n_countries=3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def test_ingest_reads_all_files(self):
        self.assertEqual(ingest.load_json_data(self.d)["country"].nunique(), 3)

    def test_train_compares_multiple_models(self):
        self.assertGreaterEqual(len(self.out["report"]["all"]["all_scores"]), 2)

    def test_predict_country_and_all(self):
        for c in ("all", "france"):
            r = model.predict(c, 2019, 3, 15, data_dir=self.d, test=True)
            self.assertIsInstance(r["y_pred"], float)

    def test_predict_unknown_country(self):
        with self.assertRaises(ValueError):
            model.predict("atlantis", 2019, 3, 15, data_dir=self.d, test=True)

    def test_features_target_horizon(self):
        ts = ingest.fetch_ts(self.d, "all")[0]
        X, y = model.engineer_features(ts)
        i = ts.index[10]
        self.assertAlmostEqual(y[i], ts["revenue"].iloc[11:41].sum())
