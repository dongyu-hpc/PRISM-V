"""Style & hygiene gate for the PRISM-V packages.

Every check here encodes a defect that a code review of this repository actually found. They
are pinned rather than fixed-once because the failure mode is silent:

* an unused import costs nothing at runtime and just accumulates
* an empty loop body (`for x in []: ...`) reads like logic and does nothing, which is worse
  than nothing -- a reader assumes the counter it feeds is populated
* a class name that violates CapWords is fine locally but breaks when the name is taught to
  a model and later renamed: the prompt, the exec namespace key and the class must agree

Scope: the PRISM-V packages only. The inherited baseline modules keep their own style and are
excluded deliberately, so the baseline arm stays byte-stable.

Run:  python tests/test_style_gate.py       (also works under pytest)
"""

from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 门禁只管新增包，基线保持原样
SCOPED_DIRS = ("prismv",)
SCOPED_FILES = ("run_prismv.py",)
MAX_COL = 100

# 已知且刻意保留的例外：PRISM_V 是对外主符号，改名要同步 prompt / exec 键 / README / 测试，
# 因此先冻结数量而不是立刻动。任何新增违规都会让这条断言失败。
CAPWORDS_EXEMPT = {"PRISM_V"}
# 允许未被直接引用的 import：类型注解字符串、re-export、带 noqa 的行
REEXPORT_MODULES = {"prismv/__init__.py", "prismv/eval/__init__.py",
                    "prismv/geover/__init__.py", "prismv/tools/__init__.py",
                    "prismv/agent/__init__.py"}


def _scoped():
    out = []
    for d in SCOPED_DIRS:
        for dp, dns, fns in os.walk(os.path.join(ROOT, d)):
            dns[:] = [x for x in dns if x != "__pycache__"]
            for fn in fns:
                if fn.endswith(".py"):
                    out.append(os.path.relpath(os.path.join(dp, fn), ROOT))
    for f in SCOPED_FILES:
        if os.path.exists(os.path.join(ROOT, f)):
            out.append(f)
    return sorted(out)


FILES = _scoped()


def _read(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read()


def test_scanned_scope_is_not_empty():
    """Guard the guard: if the walk breaks, every other test here passes vacuously."""
    assert len(FILES) >= 20, FILES
    assert "prismv/geover/predicates.py" in FILES, FILES[:5]
    print(f"  扫描范围 {len(FILES)} 个文件（仅新增包，基线不纳管）")


def test_no_unused_imports():
    offenders = []
    for p in FILES:
        src = _read(p)
        tree = ast.parse(src)
        lines = src.splitlines()
        imported = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imported[a.asname or a.name.split(".")[0]] = n.lineno
            elif isinstance(n, ast.ImportFrom):
                if n.module == "__future__":
                    continue
                for a in n.names:
                    if a.name == "*":
                        continue
                    imported[a.asname or a.name] = n.lineno
        used = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)}
        used |= {x.attr for x in ast.walk(tree) if isinstance(x, ast.Attribute)}
        # 注解常写成字符串（前向引用），一并算作使用
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Constant):
                used |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(node.annotation.value)))
            if isinstance(node, ast.FunctionDef):
                for d in node.decorator_list:
                    used |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ast.unparse(d)))
        cnt = {}
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src):
            cnt[tok] = cnt.get(tok, 0) + 1
        for name, ln in imported.items():
            if name in used:
                continue
            if ln <= len(lines) and "noqa" in lines[ln - 1]:
                continue
            if p in REEXPORT_MODULES:          # 包 __init__ 的重导出是刻意的
                continue
            if "__all__" in src and re.search(rf"['\"]{re.escape(name)}['\"]", src):
                continue                        # 显式再导出
            if cnt.get(name, 0) <= 1:
                offenders.append(f"{p}:{ln} {name}")
    assert not offenders, "未使用的 import: " + ", ".join(offenders[:10])
    print(f"  无未使用 import（{len(FILES)} 文件）")


def test_no_dead_statements():
    """空循环 / 恒空迭代 / 自我抵消的赋值：看着像逻辑，其实什么都不做。"""
    bad = []
    for p in FILES:
        lines = _read(p).splitlines()
        for i, ln in enumerate(lines, 1):
            s = ln.strip()
            if re.match(r"^for\s+\w+\s+in\s+\[\]\s*:", s):
                bad.append(f"{p}:{i} 空迭代 {s[:50]}")
            elif re.match(r"^for\s+.+:\s*$", s) and i < len(lines) \
                    and lines[i].strip() == "pass":
                bad.append(f"{p}:{i} 循环体仅 pass: {s[:50]}")
            elif re.match(r"^\w+\s*=\s*\[\]\s*if\s+.+\s+else\s+\[\]\s*$", s):
                bad.append(f"{p}:{i} 恒空赋值 {s[:50]}")
            elif re.search(r"\bif\s+False\b", s) and not s.startswith("#"):
                bad.append(f"{p}:{i} 死分支 {s[:50]}")
    assert not bad, "死语句复现:\n    " + "\n    ".join(bad)
    print("  无死循环 / 死分支 / 恒空赋值")


def test_line_length_within_budget():
    over = []
    for p in FILES:
        for i, ln in enumerate(_read(p).splitlines(), 1):
            if len(ln) > MAX_COL:
                over.append(f"{p}:{i} ({len(ln)})")
    assert not over, f"超过 {MAX_COL} 列的行: " + ", ".join(over[:8])
    print(f"  全部行 <= {MAX_COL} 列")


def test_public_class_names_are_capwords():
    offenders = []
    for p in FILES:
        for n in ast.walk(ast.parse(_read(p))):
            if isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
                if not re.match(r"^[A-Z][A-Za-z0-9]*$", n.name):
                    offenders.append(n.name)
    extra = set(offenders) - CAPWORDS_EXEMPT
    assert not extra, f"新增类名违反 CapWords: {sorted(extra)}"
    assert set(offenders) <= CAPWORDS_EXEMPT
    print(f"  公开类名合规（已冻结例外 {len(CAPWORDS_EXEMPT)} 个: "
          f"{sorted(CAPWORDS_EXEMPT)}）")


def test_no_legacy_project_name_in_new_packages():
    """新增包里不许再出现被弃用的项目名（标识符例外需在此处显式登记）。"""
    legacy = "py" + "Spatial"
    hits = []
    for p in FILES:
        for i, ln in enumerate(_read(p).splitlines(), 1):
            if legacy.lower() in ln.lower():
                hits.append(f"{p}:{i}")
    assert not hits, "出现弃用命名: " + ", ".join(hits[:8])
    print("  新增包无任何弃用命名")


def test_docstring_floor_on_public_api():
    """下限检查而非全覆盖要求：防止新模块裸奔，同时不阻塞小工具函数。

    评审记录显示新增包公开符号 docstring 覆盖约 59%，低于继承基线的 86%。一次性补齐会淹没
    真正的功能改动，所以先把地板定在 55% 并要求"每个模块至少有一个模块级 docstring"，之后
    只允许往上走。
    """
    tot = has = 0
    missing_mod = []
    for p in FILES:
        src = _read(p)
        mod = ast.parse(src)
        if not ast.get_docstring(mod):
            missing_mod.append(p)
        for n in ast.walk(mod):
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                if n.name.startswith("_") or n.name == "main":
                    continue
                tot += 1
                if ast.get_docstring(n):
                    has += 1
    ratio = has / max(1, tot)
    assert not missing_mod, "缺少模块级 docstring: " + ", ".join(missing_mod)
    assert ratio >= 0.55, f"docstring 覆盖率 {ratio:.2f} 低于地板 0.55"
    print(f"  模块 docstring 全覆盖；公开符号覆盖 {has}/{tot} = {ratio*100:.0f}%（地板 55%）")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    print(f"running {len(ALL_TESTS)} style/hygiene gate tests\n")
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as e:
            failed.append((fn.__name__, str(e)))
            print(f"  !! {fn.__name__} FAILED: {e}")
        except Exception as e:
            failed.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"  !! {fn.__name__} ERROR: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
