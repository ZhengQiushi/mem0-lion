import os
from mem0 import Memory
from faker import Faker
import random
import json

os.environ["GOOGLE_API_KEY"] = "AIzaSyDLZHwabgwDySAeE9GaUb1kJEM4VgkiImc"

config = {
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "gemini-embedding-001",
            "embedding_dims": 1536,
        }
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-2.5-flash",
        }
    },
    "vector_store": {
        "provider": "elasticsearch",
        "config": {
            "collection_name": "mem_test_mocked",
            "host": "https://ec6d45523ca143c38f43de6dd9257d94.westus2.azure.elastic-cloud.com",
            "port": 443,
            "embedding_model_dims": 1536,
            "api_key": "b0lCNW5aY0IyOUdoQ0VUVVVFSlY6OExqekM3MTZTV21PS2VUb3VNaV9XZw==",
        }
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "xxx"
        }
    }
}

m = Memory.from_config(config)
fake = Faker()
user_id = "test_user_memory_manager_optimized" # 使用不同的user_id以避免冲突

def print_current_memory(memory_instance, step_name):
    print(f"\n--- Memory after {step_name} ---")
    all_memories = memory_instance.get_all(user_id=user_id)
    if all_memories:
        print(json.dumps(all_memories, indent=2))
    else:
        print("Memory is empty.")
    print("-" * 30)

def test_memory_retrieval_and_context(memory_instance, user_id):
    print("\n--- Testing Memory Retrieval and Context ---")

    # --- Test Case 1: Forgetting older information (simulated by search context) ---
    # We want to see if the search prioritizes recent information and can still
    # correctly infer the current location when older location information exists.
    # The `search` function in mem0 by default retrieves based on relevance.
    # If it's configured to have a limited history window or if the new information
    # strongly supersedes old information, then older context might be "forgotten" in practice.

    print("\n--- Scenario 1: Testing context retention for location ---")
    # Current last turn is about health and snacks in New York.
    # Let's ask a question about supermarkets in the *current* location,
    # simulating that the "New Jersey" information is no longer the primary context.

    query_1 = "Where can I find good supermarkets?"
    print(f"\nQuery: '{query_1}'")
    # The search should retrieve memories relevant to "supermarkets" and consider the most recent context.
    # The most recent context has established the user is in New York.
    # We expect the LLM to infer "near me" refers to New York.
    search_results_1 = memory_instance.search(query_1, user_id=user_id)
    print(f"Search Results for '{query_1}':")
    print(json.dumps(search_results_1, indent=2))
    # Expected: Results should point to supermarkets in New York, and potentially reference the health context.
    # The "New Jersey" memories should be less relevant for a "near me" query in the current context.

    query_2 = "What about movie theaters?"
    print(f"\nQuery: '{query_2}'")
    # Similar to above, the LLM should understand "near me" as the current location (New York).
    search_results_2 = memory_instance.search(query_2, user_id=user_id)
    print(f"Search Results for '{query_2}':")
    print(json.dumps(search_results_2, indent=2))
    # Expected: Results should be about movie theaters in New York.

    # --- Test Case 2: Incorrect information leading to misdirection ---
    # The false information "I actually hate pizza" was added at Step 10 of population.
    # We need to query something that would be affected by this false preference.

    print("\n--- Scenario 2: Testing misdirection from incorrect information ---")
    # Query for food recommendations, where the "hate pizza" false memory might influence the result.
    query_3 = "What's a good dinner option for tonight?"
    print(f"\nQuery: '{query_3}'")
    search_results_3 = memory_instance.search(query_3, user_id=user_id)
    print(f"Search Results for '{query_3}':")
    print(json.dumps(search_results_3, indent=2))
    # Expected: The retrieved memories should include the false statement "I actually hate pizza."
    # The LLM's response (if it were to generate one based on these results) would likely
    # avoid recommending pizza, demonstrating the misdirection.

    # Now, let's see if the correction works.
    print("\n--- Correcting the false information and re-querying ---")
    # We need to add the correction first, as this script only *searches*.
    # In a real interaction, the user would provide this correction.
    # For this script's purpose, we'll simulate adding the correction to the memory.
    print("\n--- Simulating adding the correction: 'I do like pizza' ---")
    conversation_correction = [
        {"role": "user", "content": "Actually, I was wrong. I do like pizza. I'd love to find a good pizza place in New York."},
        {"role": "assistant", "content": "My apologies! It's good to know you enjoy pizza. Let me find some great pizza places for you in New York."}
    ]
    memory_instance.add(conversation_correction, user_id=user_id)
    print("Correction added.")
    print_current_memory(memory_instance, "After Adding Correction")

    # Re-query for dinner options.
    print(f"\nQuery: '{query_3}' (after correction)")
    search_results_4 = memory_instance.search(query_3, user_id=user_id)
    print(f"Search Results for '{query_3}' (after correction):")
    print(json.dumps(search_results_4, indent=2))
    # Expected: The search results should now reflect that the user *likes* pizza,
    # and the false memory of "hating pizza" should be less prominent or superseded
    # by the newer, corrected information.

    # To further test the forgetting aspect in relation to the correction:
    # Ask about something specific in New York, and see if the LLM remembers the "pizza place in New York" recommendation.
    query_4 = "What's a good pizza place in New York?"
    print(f"\nQuery: '{query_4}'")
    search_results_5 = memory_instance.search(query_4, user_id=user_id)
    print(f"Search Results for '{query_4}':")
    print(json.dumps(search_results_5, indent=2))
    # Expected: The search results should be highly relevant to finding pizza places in New York,
    # indicating that the corrected preference and location are prioritized.


# --- Main Execution ---
print("--- Starting Memory Search Test Scenario ---")

# Run the search and context testing
test_memory_retrieval_and_context(m, user_id)

print("\n--- Memory Search Test Scenario Completed ---")