"""
Category extraction tools for building category subgraphs.
Based on extract_category.py logic but adapted for graph memory system.
"""

from typing import Dict, List, Any
from mem0.graphs.extract_category import get_prompt, get_kwargs, pack_input, get_default_profiles


# Tool for extracting categories from user data
EXTRACT_CATEGORIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_categories",
        "description": "Extract user profile categories and subcategories from text data. This creates a hierarchical structure: user -> topic -> sub_topic -> memo.",
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string", 
                                "description": "The main topic category (e.g., 'interest', 'work', 'basic_info')"
                            },
                            "sub_topic": {
                                "type": "string", 
                                "description": "The specific subcategory (e.g., 'movie', 'title', 'name')"
                            },
                            "memo": {
                                "type": "string", 
                                "description": "The detailed information or fact about this category"
                            }
                        },
                        "required": ["topic", "sub_topic", "memo"],
                        "additionalProperties": False,
                    },
                    "description": "An array of category information extracted from the text."
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    },
}

# Structured version for structured LLM providers
EXTRACT_CATEGORIES_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_categories",
        "description": "Extract user profile categories and subcategories from text data. This creates a hierarchical structure: user -> topic -> sub_topic -> memo.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string", 
                                "description": "The main topic category (e.g., 'interest', 'work', 'basic_info')"
                            },
                            "sub_topic": {
                                "type": "string", 
                                "description": "The specific subcategory (e.g., 'movie', 'title', 'name')"
                            },
                            "memo": {
                                "type": "string", 
                                "description": "The detailed information or fact about this category"
                            }
                        },
                        "required": ["topic", "sub_topic", "memo"],
                        "additionalProperties": False,
                    },
                    "description": "An array of category information extracted from the text."
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    },
}

# Tool for query category classification
CLASSIFY_QUERY_CATEGORIES_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_query_categories",
        "description": "Classify a user query to identify which categories might be relevant for context-aware retrieval.",
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string", 
                                "description": "The main topic category that might be relevant to the query"
                            },
                            "sub_topic": {
                                "type": "string", 
                                "description": "The specific subcategory that might be relevant"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score (0-1) for this category being relevant to the query"
                            }
                        },
                        "required": ["topic", "sub_topic", "confidence"],
                        "additionalProperties": False,
                    },
                    "description": "Categories that might be relevant for the query"
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    },
}

# Structured version for query classification
CLASSIFY_QUERY_CATEGORIES_STRUCT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_query_categories",
        "description": "Classify a user query to identify which categories might be relevant for context-aware retrieval.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string", 
                                "description": "The main topic category that might be relevant to the query"
                            },
                            "sub_topic": {
                                "type": "string", 
                                "description": "The specific subcategory that might be relevant"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score (0-1) for this category being relevant to the query"
                            }
                        },
                        "required": ["topic", "sub_topic", "confidence"],
                        "additionalProperties": False,
                    },
                    "description": "Categories that might be relevant for the query"
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    },
}


def get_category_extraction_prompt(topic_examples: str = None) -> str:
    """Get the prompt for category extraction."""
    if topic_examples is None:
        topic_examples = get_default_profiles()
    return get_prompt(topic_examples)


def get_category_extraction_kwargs() -> Dict[str, Any]:
    """Get additional kwargs for category extraction."""
    return get_kwargs()


def pack_category_input(already_input: str, memo_str: str, strict_mode: bool = False) -> str:
    """Pack input for category extraction."""
    return pack_input(already_input, memo_str, strict_mode)
