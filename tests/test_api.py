import shutil, tempfile, unittest
from app import app
from tests.helpers import make_synthetic_data

class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        make_synthetic_data(cls.d)
        cls.c = app.test_client()
        r = cls.c.post("/train", json={"test": True, "data_dir": cls.d})
        assert r.status_code == 200, r.get_json()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def post(self, **kw):
        return self.c.post("/predict", json={"test": True, "data_dir": self.d, **kw})

    def test_predict_country(self):
        r = self.post(country="france", year=2019, month=2, day=28)
        self.assertEqual(r.status_code, 200)
        self.assertIn("y_pred", r.get_json())

    def test_predict_all(self):
        self.assertEqual(self.post(year=2019, month=2, day=15).status_code, 200)

    def test_missing_fields(self):
        self.assertEqual(self.c.post("/predict", json={"test": True}).status_code, 400)

    def test_bad_country(self):
        self.assertEqual(self.post(country="zzz", year=2019, month=2, day=15).status_code, 400)

    def test_logs_endpoint(self):
        self.post(year=2019, month=2, day=15)
        names = self.c.get("/logs?test=true").get_json()
        self.assertTrue(any(n.startswith("predict") for n in names))
        self.assertEqual(self.c.get("/logs/nope.log?test=true").status_code, 404)
