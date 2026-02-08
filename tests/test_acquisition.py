import unittest

from pipeline.acquisition import _doc_fetch_is_valid


class AcquisitionHelpersTest(unittest.TestCase):
    def test_doc_fetch_valid_requires_success_and_pages(self):
        self.assertTrue(_doc_fetch_is_valid({"success": True, "pages": 3}))
        self.assertFalse(_doc_fetch_is_valid({"success": True, "pages": 0}))
        self.assertFalse(_doc_fetch_is_valid({"success": False, "pages": 10}))


if __name__ == "__main__":
    unittest.main()
