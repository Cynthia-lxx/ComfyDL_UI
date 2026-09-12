#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ComfyDL 节点语言包生成脚本 (node translation pack generator)
================================================================

从 ComfyDL 注册表与中文文档实时生成仓库级语言包 `locales/zh/nodeDefs.json`：

  * `display_name`：取 `comfydl/FUNCTIONS_zh.md` 中该节点的中文小节标题；标题仍是英文
    （Area Chart、ResNet-18 等术语）的节点不生成该字段，前端自动回退到注册表中的英文名。
  * `description`：取同一小节的 `- **功能**：` 一行，去掉 Markdown 标记后作为节点说明。

英文侧不生成：注册表里的名字与 docstring 本身就是英文，语言包只在有译文时才有意义。

用法:
    python cdl_smoke_tests/gen_locales.py           # 写入语言包
    python cdl_smoke_tests/gen_locales.py --check   # 只校验，不写文件；有差异退出码 1
退出码:
    0 = 无差异（或已同步）；1 = check 模式存在差异 / 执行失败
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # 不写入 .pyc

_ROOT = Path(__file__).resolve().parent.parent          # ComfyDL_UI/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import comfydl  # noqa: E402  触发插件入口注册

_DOC = _ROOT / "comfydl" / "FUNCTIONS_zh.md"
_OUT = _ROOT / "locales" / "zh" / "nodeDefs.json"

# 一个中文小节：标题行 + 紧随其后的 `- **类名**：` 行，直到下一个小节。
_ENTRY_RE = re.compile(
    r"^### (.+?)\n- \*\*类名\*\*：`(\w+)`(.*?)(?=^### |\Z)", re.M | re.S
)
_FUNC_RE = re.compile(r"^- \*\*功能\*\*：(.+)$", re.M)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def clean(text):
    """去掉 Markdown 强调/行内代码标记，压平空白。"""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "")).strip()


def parse_doc():
    """从中文文档解析出 {node_id: {display_name?, description?}}。"""
    entries = {}
    for title, node_id, body in _ENTRY_RE.findall(_DOC.read_text(encoding="utf-8")):
        entry = {}
        if _CJK_RE.search(title):
            entry["display_name"] = clean(title)
        function = _FUNC_RE.search(body)
        if function:
            entry["description"] = clean(function.group(1))
        if entry:
            entries[node_id] = entry
    return entries


def build():
    """生成 {node_id: {...}}，只保留仍在注册表中的节点。"""
    documented = parse_doc()
    registered = sorted(comfydl.NODE_CLASS_MAPPINGS)
    pack = {node_id: documented[node_id] for node_id in registered if node_id in documented}

    missing = [node_id for node_id in registered if node_id not in documented]
    if missing:
        print(f"未在 FUNCTIONS_zh.md 中找到：{len(missing)} 个节点 {missing[:5]}...")
    return dict(sorted(pack.items()))


def serialize(pack):
    return json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成 locales/zh/nodeDefs.json")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args(argv)

    content = serialize(build())
    if args.check:
        current = _OUT.read_text(encoding="utf-8") if _OUT.exists() else ""
        if current != content:
            print(f"{_OUT.relative_to(_ROOT)} 与文档不一致，请运行 gen_locales.py")
            return 1
        print(f"{_OUT.relative_to(_ROOT)} 已是最新")
        return 0

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(content, encoding="utf-8")
    print(f"已写入 {_OUT.relative_to(_ROOT)}：{len(json.loads(content))} 个节点条目")
    return 0


if __name__ == "__main__":
    sys.exit(main())
