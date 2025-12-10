#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil, os
path = "rag/chroma"
if os.path.isdir(path):
    shutil.rmtree(path)
    print(f"[OK] 已刪除 {path}")
else:
    print(f"[SKIP] 未找到 {path}")
os.makedirs(path, exist_ok=True)
