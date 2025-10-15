#!/usr/bin/env python3
"""
简化的测试脚本，验证category子图功能
"""

import sys
sys.path.insert(0, '/home/azureuser/mem0')

from mem0.graphs.category_tools import (
    EXTRACT_CATEGORIES_TOOL,
    get_category_extraction_prompt,
    get_default_profiles
)

def test_category_tools():
    """测试category工具"""
    print("=== 测试Category工具 ===")
    
    # 测试工具定义
    print("1. 检查工具定义:")
    print(f"   EXTRACT_CATEGORIES_TOOL: {EXTRACT_CATEGORIES_TOOL['function']['name']}")
    
    # 测试prompt生成
    print("\n2. 测试prompt生成:")
    try:
        prompt = get_category_extraction_prompt()
        print(f"   Prompt长度: {len(prompt)} 字符")
        print(f"   Prompt前100字符: {prompt[:100]}...")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试默认profiles
    print("\n3. 测试默认profiles:")
    try:
        profiles = get_default_profiles()
        print(f"   Profiles长度: {len(profiles)} 字符")
        print(f"   Profiles前100字符: {profiles[:100]}...")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_category_tools()
