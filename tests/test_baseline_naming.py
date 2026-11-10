"""Static guard for the project's single naming scheme.

A rename across a code-generation system is exactly the change that passes every import
check and still breaks at runtime: the class, the module, the ``exec`` namespace key injected
for generated programs, and the name taught in the prompt are four separate sites that must
agree. If the prompt teaches one spelling and the namespace provides another, no import fails
-- generated programs raise NameError inside ``exec``, the evaluation loop records a failed
parse, and the accuracy number quietly absorbs the loss.

So the invariants are checked statically, and checked against the *prompt text*, which is the
site nobody remembers to update.

Note on the literal: the legacy name is assembled from fragments rather than written out, so
this guard file does not itself become a hit when it scans the tree.

Run:  python tests/test_baseline_naming.py     (also works under pytest)
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LEGACY = "py" + "Spatial"                    # the retired project name
LEGACY_MODULE = LEGACY + "_Interface"        # the retired module name
API = "SpatialAPI"                           # current baseline facade class
MODULE = "baseline_interface"

TOOLS = ("reconstruct", "describe_camera_motion", "synthesize_novel_view",
         "rotate_right", "rotate_left", "move_forward", "move_backward", "turn_around")

SKIP_DIRS = {"base_model", "__pycache__", ".pytest_cache", ".git"}


def _read(rel):
    return io.open(os.path.join(ROOT, rel), encoding="utf-8").read()


def _tracked_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith((".py", ".md", ".txt", ".yaml", ".yml", ".gitignore")):
                out.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    return sorted(out)


# --------------------------------------------------------------------------- absence
def test_legacy_name_absent_everywhere():
    hits = []
    for rel in _tracked_files():
        text = _read(rel)
        for i, line in enumerate(text.splitlines(), 1):
            if LEGACY.lower() in line.lower():
                hits.append(f"{rel}:{i}")
    assert not hits, f"legacy name still present in {len(hits)} place(s): " + ", ".join(hits[:12])
    print(f"  全树无遗留命名（扫描 {len(_tracked_files())} 个文件）")


def test_legacy_module_file_gone():
    assert os.path.exists(os.path.join(ROOT, MODULE + ".py")), f"{MODULE}.py 不存在"
    assert not os.path.exists(os.path.join(ROOT, LEGACY_MODULE + ".py")), "旧模块文件仍在仓库里"
    print(f"  模块文件: {MODULE}.py 就位，旧文件已移除")


# --------------------------------------------------------------------------- surface
def test_baseline_interface_defines_the_expected_surface():
    tree = ast.parse(_read(MODULE + ".py"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    for needed in ("Reconstruction", "Scene", API, "Agent"):
        assert needed in classes, f"{MODULE} 缺少类 {needed}"

    methods = {m.name for m in classes[API].body if isinstance(m, ast.FunctionDef)}
    missing = set(TOOLS) - methods
    assert not missing, f"{API} 缺少工具方法: {sorted(missing)}"

    # the class-level cache directory is a documented global; losing it silently disables
    # every preprocessed scene and forces live reconstruction
    assigned = {t.id for a in classes[API].body if isinstance(a, ast.Assign)
                for t in a.targets if isinstance(t, ast.Name)}
    assert "PROCESSED_BASE_DIR" in assigned, "PROCESSED_BASE_DIR 全局配置点丢失"

    agent_methods = {m.name for m in classes["Agent"].body if isinstance(m, ast.FunctionDef)}
    for needed in ("generate_code", "parse_LLM_response", "execute", "answer", "basic_qa"):
        assert needed in agent_methods, f"Agent.{needed} 丢失"
    print(f"  {API} {len(methods)} 个工具方法 + Agent {len(agent_methods)} 个编排方法齐备")


# ------------------------------------------------------- prompt / namespace agreement
def test_generated_program_namespace_matches_prompt():
    ex = _read(os.path.join("agent", "codeAgent", "execute.py"))
    assert f"from {MODULE} import" in ex, "execute.py 未从改名后的模块导入"

    m = re.search(r"execution_globals\s*=\s*\{(.*?)\}", ex, re.S)
    assert m, "找不到 execution_globals 字面量"
    keys = set(re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:", m.group(1)))
    assert API in keys, f"exec 命名空间未注入 {API}: {sorted(keys)}"
    assert LEGACY not in keys, f"exec 命名空间仍暴露旧名: {sorted(keys)}"
    assert {"Scene", "Reconstruction"} <= keys, "生成程序需要的数据类型未注入"

    prompt = _read(os.path.join("agent", "prompt", "template.py"))
    taught = sorted({meth for _, meth in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.(%s)\b" % "|".join(TOOLS), prompt)})
    assert taught, "prompt 里没有任何工具调用示例，模板可能被改坏"
    taught_objs = {obj for obj, _ in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.(%s)\b" % "|".join(TOOLS), prompt)}
    assert taught_objs == {API}, (
        f"prompt 教的调用名 ({sorted(taught_objs)}) 与 exec 注入键 ({API}) 不一致 -- "
        "生成程序会在 exec 内 NameError，表现为 parse 失败而非导入错误")

    # every taught method must actually exist
    tree = ast.parse(_read(MODULE + ".py"))
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == API][0]
    have = {mm.name for mm in cls.body if isinstance(mm, ast.FunctionDef)}
    assert set(taught) <= have, f"prompt 教了不存在的工具: {sorted(set(taught) - have)}"
    print(f"  prompt 教的 {len(taught)} 个调用全部有实现，注入键一致")


def test_prismv_prompt_matches_prismv_namespace():
    """Same invariant on the PRISM-V arm: template names must equal namespace keys."""
    ns_src = _read(os.path.join("prismv", "pipeline.py"))
    m = re.search(r"ns\s*=\s*\{(.*?)\}", ns_src, re.S)
    assert m, "找不到 build_execution_namespace 的 ns 字面量"
    keys = set(re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:", m.group(1)))
    assert "PRISM_V" in keys, f"PRISM-V 执行命名空间缺 PRISM_V: {sorted(keys)}"
    assert LEGACY not in keys, "PRISM-V 执行命名空间不应再暴露历史名"

    tmpl = _read(os.path.join("prismv", "agent", "template.py"))
    objs = {o for o, _ in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.(verify|check_options|rank_relations|ground_entities|"
        r"reconstruct|list_entities|verifiable_relations|scene_statistics)\b", tmpl)}
    assert objs <= {"PRISM_V"}, f"PRISM-V prompt 出现了命名空间之外的对象名: {sorted(objs)}"
    print(f"  PRISM-V prompt 只用 {sorted(objs)}，与命名空间一致")


def test_prismv_exposes_only_prism_v():
    src = _read(os.path.join("prismv", "interface.py"))
    tree = ast.parse(src)
    assert any(isinstance(n, ast.ClassDef) and n.name == "PRISM_V" for n in tree.body), \
        "PRISM_V 类定义丢失"
    module_aliases = {t.id for n in tree.body if isinstance(n, ast.Assign)
                      for t in n.targets if isinstance(t, ast.Name)}
    assert LEGACY not in module_aliases, f"prismv 仍在导出历史别名: {LEGACY}"
    allv = re.search(r"__all__\s*=\s*\[(.*?)\]", src, re.S)
    assert allv and LEGACY not in allv.group(1), f"__all__ 仍导出 {LEGACY}"
    print("  PRISM-V 只导出 PRISM_V / PRISMV，无历史别名")


# --------------------------------------------------------------------------- imports
def test_every_import_site_points_at_renamed_module():
    legacy_pat = re.compile(r"\b" + re.escape(LEGACY_MODULE) + r"\b")
    sites = []
    for rel in _tracked_files():
        if not rel.endswith(".py"):
            continue
        text = _read(rel)
        if legacy_pat.search(text):
            sites.append(rel)
        if f"import {MODULE}" in text or f"from {MODULE}" in text:
            # the module must define whatever is imported
            tree = ast.parse(_read(MODULE + ".py"))
            exported = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
            exported |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef,))}
            for names in re.findall(rf"from {MODULE} import ([^\n(]+)", text):
                for nm in [x.strip() for x in names.split(",")]:
                    nm = nm.split(" as ")[0].strip()
                    if nm and nm != "*":
                        assert nm in exported, f"{rel} 从 {MODULE} 导入了不存在的 {nm}"
    assert not sites, f"仍有文件引用旧模块名: {sites}"
    print("  所有 import 站点指向改名后的模块，且导入符号都存在")


def test_entry_points_still_configured_correctly():
    mc = _read("mindcube.py")
    assert f"from {MODULE} import Agent, Scene, {API}" in mc, "mindcube 主导入未更新"
    assert f"{API}.PROCESSED_BASE_DIR" in mc, "mindcube 全局缓存目录赋值未更新"
    # spawn workers do not inherit parent globals; the re-apply line must survive the rename
    assert re.search(rf"{API}\.PROCESSED_BASE_DIR\s*=\s*processed_dir", mc), \
        "子进程重设缓存目录的那行丢失，会导致每张图现场重跑重建"
    cli = _read("run_prismv.py")
    assert LEGACY not in cli, "CLI 文档串里仍有历史命名"
    print("  mindcube 基线入口与 CLI 的全局配置路径一致")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    print(f"running {len(ALL_TESTS)} naming-consistency tests\n")
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
