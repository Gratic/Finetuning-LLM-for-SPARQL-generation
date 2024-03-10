from pathlib import Path
from scripts.evaluation_bench import main, create_parser
import unittest

class EvaluationBenchTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)    
        
    def test_main_with_evaluation_test(self):
        filepath = Path("tests/scriptstests/outputs/EvaluationBenchTest.test_main_with_evaluation_test.json")
        
        if filepath.exists():
            filepath.unlink()
        
        parser = create_parser()
        
        args = parser.parse_args([
            "--dataset=tests/scriptstests/data/evaluation_test/evaluation_test.json",
            "--preprocess-gold=tests/scriptstests/data/evaluation_test/evaluation_test_gold.json",
            "--model=foo",
            f"--output={str(filepath.parent)}",
            f"--save-name={str(filepath.stem)}",
        ])
        
        main(args)
        
        self.assertTrue(filepath.exists())
        
        if filepath.exists():
            filepath.unlink()