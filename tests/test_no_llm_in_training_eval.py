import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_stage1_no_llm_import():
    content = (Path(__file__).parent.parent / "scripts" / "train_stage1.py").read_text()
    assert "from evidence.llm_teacher" not in content
    assert "import llm_teacher" not in content


def test_train_stage3_no_llm_import():
    content = (Path(__file__).parent.parent / "scripts" / "train_stage3.py").read_text()
    assert "from evidence.llm_teacher" not in content
    assert "import llm_teacher" not in content


def test_evaluate_no_llm_import():
    content = (Path(__file__).parent.parent / "scripts" / "evaluate.py").read_text()
    assert "from evidence.llm_teacher" not in content
    assert "import llm_teacher" not in content


def test_models_no_llm_import():
    for py_file in (Path(__file__).parent.parent / "models").glob("*.py"):
        content = py_file.read_text()
        assert "from evidence.llm_teacher" not in content
        assert "import llm_teacher" not in content


def test_training_no_llm_import():
    for py_file in (Path(__file__).parent.parent / "training").glob("*.py"):
        content = py_file.read_text()
        assert "from evidence.llm_teacher" not in content
        assert "import llm_teacher" not in content
