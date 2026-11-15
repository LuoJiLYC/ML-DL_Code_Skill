"""Validate experiment structure and new-code issues relative to canonical templates."""
import argparse
import ast
import difflib
import io
import tokenize
from pathlib import Path

REQUIRED_ML_FUNCS = {"preprocess_data", "initialize_model", "train_model", "test_model", "main"}
REQUIRED_DL_FOLD_FUNCS = {"preprocess_data", "initialize_model", "main"}
MODEL_NAMES = ["logistic_regression", "catboost", "lightgbm", "xgboost", "gbdt", "knn", "svm", "rf", "dt", "mlp"]
FOLD_DIRS = {"ml_cls_fold", "dl_cls_fold"}


def template_for(relative, skill_root, new_path):
    candidates = []
    template = skill_root / "scripts" / relative
    if template.is_file():
        candidates.append(template)

    if relative.parts[0] in {"ml_cls", "dl_cls", "ml_cls_fold", "dl_cls_fold"} and relative.suffix == ".py":
        name = relative.stem
        for model in MODEL_NAMES:
            if name == model or name.startswith(model + "_"):
                candidate = skill_root / "scripts" / relative.parts[0] / (model + ".py")
                if candidate.is_file():
                    candidates.append(candidate)

    for exp in ["exp1_random_split", "exp2_fixed_split", "exp3_5-fold"]:
        candidate = skill_root / "assets" / exp / "code" / relative
        if candidate.is_file():
            candidates.append(candidate)

    if not candidates:
        return None

    new_lines = new_path.read_text(encoding="utf-8-sig").splitlines()
    return max(candidates, key=lambda c: difflib.SequenceMatcher(
        a=c.read_text(encoding="utf-8-sig").splitlines(), b=new_lines, autojunk=False
    ).ratio())


def changed_lines(new_lines, base_lines):
    if base_lines is None:
        return set(range(1, len(new_lines) + 1))
    changed = set()
    matcher = difflib.SequenceMatcher(a=base_lines, b=new_lines, autojunk=False)
    for tag, _, _, b1, b2 in matcher.get_opcodes():
        if tag != "equal":
            changed.update(range(b1 + 1, b2 + 1))
    return changed


def syntax_and_style(path, canonical):
    errors = []
    try:
        new_text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(new_text, filename=str(path))
    except (UnicodeError, SyntaxError) as exc:
        return None, [f"{path}: Python 语法或编码错误：{exc}"]

    lines = new_text.splitlines()
    base_lines = canonical.read_text(encoding="utf-8-sig").splitlines() if canonical else None
    new_line_numbers = changed_lines(lines, base_lines)

    for line_number in sorted(new_line_numbers):
        if len(lines[line_number - 1]) > 120:
            errors.append(f"{path}:{line_number}：新增或改动的代码行超过 120 字符")

    try:
        for token in tokenize.generate_tokens(io.StringIO(new_text).readline):
            if token.type == tokenize.OP and token.string == ";" and token.start[0] in new_line_numbers:
                errors.append(f"{path}:{token.start[0]}：新增代码中使用分号连接语句")
    except tokenize.TokenError as exc:
        errors.append(f"{path}：Token 解析失败：{exc}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno in new_line_numbers:
            if node.body and node.body[0].lineno == node.lineno:
                errors.append(f"{path}:{node.lineno}：新增单行函数")
    return tree, errors


def check_no_runpy(file_path, tree):
    errors = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "runpy" for alias in node.names):
            errors.append(f"{file_path}：不允许 runpy 转发模型入口")
        if isinstance(node, ast.ImportFrom) and node.module == "runpy":
            errors.append(f"{file_path}：不允许 runpy 转发模型入口")
    return errors


def check_ml_model(file_path, tree):
    funcs = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = REQUIRED_ML_FUNCS - funcs
    errors = []
    if missing:
        errors.append(f"{file_path}：传统 ML 模型缺少函数：{', '.join(sorted(missing))}")
    return errors + check_no_runpy(file_path, tree)


def check_dl_fold_model(file_path, tree):
    funcs = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = REQUIRED_DL_FOLD_FUNCS - funcs
    errors = []
    if missing:
        errors.append(f"{file_path}：DL Fold 模型缺少函数：{', '.join(sorted(missing))}")
    imported_splitter = any(
        isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("cross_validation.splitter")
        for node in tree.body
    )
    if not imported_splitter:
        errors.append(f"{file_path}：DL Fold 模型未导入公共 cross_validation.splitter")
    return errors + check_no_runpy(file_path, tree)


def main():
    parser = argparse.ArgumentParser(description="检查新实验代码，按母版基线豁免既有风格")
    parser.add_argument("--experiment", type=Path, required=True, help="含 code/ 的实验根目录")
    parser.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    skill_root = args.skill_root.resolve()
    code_dir = experiment / "code"
    errors = []

    if not code_dir.is_dir():
        errors.append(f"缺少实验代码目录：{code_dir}")
    else:
        required_dirs = ["common", "plotting"]
        model_dirs = ["ml_cls", "dl_cls", "ml_cls_fold", "dl_cls_fold"]
        if not any((code_dir / name).is_dir() for name in model_dirs):
            errors.append("缺少模型目录：至少需要 ml_cls/、dl_cls/、ml_cls_fold/、dl_cls_fold/ 中的一个")
        for sub_dir in required_dirs:
            if not (code_dir / sub_dir).is_dir():
                errors.append(f"缺少规定目录：{code_dir / sub_dir}")

        fold_present = any((code_dir / name).is_dir() for name in FOLD_DIRS)
        if fold_present and not (code_dir / "common" / "cross_validation" / "splitter.py").is_file():
            errors.append("Fold 模型目录存在，但缺少 common/cross_validation/splitter.py")

        python_files = sorted(code_dir.rglob("*.py"))
        if not python_files:
            errors.append("实验目录中没有 Python 文件")

        for path in python_files:
            relative = path.relative_to(code_dir)
            canonical = template_for(relative, skill_root, path)
            tree, issues = syntax_and_style(path, canonical)
            errors.extend(issues)
            if tree is None or path.name == "__init__.py":
                continue
            if len(relative.parts) == 2 and relative.parts[0] == "ml_cls":
                errors.extend(check_ml_model(relative, tree))
            elif len(relative.parts) == 2 and relative.parts[0] == "ml_cls_fold":
                errors.extend(check_ml_model(relative, tree))
            elif len(relative.parts) == 2 and relative.parts[0] == "dl_cls_fold":
                errors.extend(check_dl_fold_model(relative, tree))
            if relative.parts[0] == "common" and relative.stem in {"classical_runner", "model_runner", "multi_model_runner"}:
                errors.append(f"{relative}：禁止集中式多模型 Runner")

    if errors:
        print(f"FAIL：发现 {len(errors)} 个问题")
        for issue in errors:
            print(" -", issue)
        return 1
    print("PASS：目录、模型入口、语法、新增严重风格问题检查通过")
    print("提示：业务语义、main() 逻辑分段及模型专属职责仍需人工/Agent 检查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
