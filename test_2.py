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
            "max_tokens": 10000,
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

def run_optimized_test_scenario(memory_instance):
    print("--- Starting Optimized Memory Manager Test Scenario ---")

    # Clear existing memories for the user_id before running the test
    print(f"Clearing memories for user_id: {user_id}")
    delete_results = memory_instance.delete_all(user_id=user_id)
    print(f"Delete results: {delete_results}")
    print("Memories cleared.")

    # Step 0: Initial location setting
    conversation_0 = [
        {"role": "user", "content": "My home is in New Jersey."},
        {"role": "assistant", "content": "Okay, New Jersey. Got it."}
    ]
    print("\n--- Initializing memory with user's home location ---")
    results_0 = memory_instance.add(conversation_0, user_id=user_id)
    print(f"Add results: {results_0}")
    print_current_memory(memory_instance, "Initial Location Add")

    # Step 1: Ask about nearby movie theaters
    conversation_1_1 = [
        {"role": "user", "content": "Are there any good movie theaters near me?"},
        {"role": "assistant", "content": "Searching for movie theaters near New Jersey..."}
    ]
    print("\n--- Asking about nearby movie theaters ---")
    results_1_1 = memory_instance.add(conversation_1_1, user_id=user_id)
    print(f"Add results: {results_1_1}")
    print_current_memory(memory_instance, "Movie Theaters Question")

    # Step 2: Ask about nearby supermarkets
    conversation_1_2 = [
        {"role": "user", "content": "What about supermarkets?"},
        {"role": "assistant", "content": "Okay, checking for supermarkets in New Jersey."}
    ]
    print("\n--- Asking about nearby supermarkets ---")
    results_1_2 = memory_instance.add(conversation_1_2, user_id=user_id)
    print(f"Add results: {results_1_2}")
    print_current_memory(memory_instance, "Supermarkets Question")

    # Step 3: Express preferences and ask for general recommendations
    conversation_2_1 = [
        {"role": "user", "content": "I like to eat potato chips and drink beer. Do you have any latest recommendations?"},
        {"role": "assistant", "content": "For potato chips, I recommend 'Fancy Brand Crisps'. For beer, 'Local Craft Brew' is popular."}
    ]
    print("\n--- Expressing preferences and asking for general recommendations ---")
    results_2_1 = memory_instance.add(conversation_2_1, user_id=user_id)
    print(f"Add results: {results_2_1}")
    print_current_memory(memory_instance, "Snacks & Beer General Recommendation")

    # Step 4: Ask for details about specific beer
    conversation_2_2 = [
        {"role": "user", "content": "Tell me more about 'Local Craft Brew' beer."},
        {"role": "assistant", "content": "'Local Craft Brew' is an IPA with citrus notes, brewed in small batches."}
    ]
    print("\n--- Asking for details about specific beer ---")
    results_2_2 = memory_instance.add(conversation_2_2, user_id=user_id)
    print(f"Add results: {results_2_2}")
    print_current_memory(memory_instance, "Specific Beer Details")

    # Step 5: Ask for similar beer recommendations
    conversation_2_3 = [
        {"role": "user", "content": "Sounds good! What other similar beers would you suggest?"},
        {"role": "assistant", "content": "If you like 'Local Craft Brew', you might also enjoy 'Hoppy Trails IPA' or 'Golden Ale'."}
    ]
    print("\n--- Asking for similar beer recommendations ---")
    results_2_3 = memory_instance.add(conversation_2_3, user_id=user_id)
    print(f"Add results: {results_2_3}")
    print_current_memory(memory_instance, "Similar Beer Recommendations")

    # Step 6: Ask about availability in current location
    conversation_2_4 = [
        {"role": "user", "content": "Are those available in New Jersey?"},
        {"role": "assistant", "content": "Yes, they should be widely available in New Jersey liquor stores and supermarkets."}
    ]
    print("\n--- Asking about availability in current location ---")
    results_2_4 = memory_instance.add(conversation_2_4, user_id=user_id)
    print(f"Add results: {results_2_4}")
    print_current_memory(memory_instance, "Availability in New Jersey")

    # Step 7: Ask for movie recommendation
    conversation_2_5 = [
        {"role": "user", "content": "Great, I'll look for them. What about a good movie to watch with these snacks?"},
        {"role": "assistant", "content": "With chips and beer, a good action-comedy or a suspense thriller would be perfect. I recommend 'The Big Heist'."}
    ]
    print("\n--- Asking for movie recommendation ---")
    results_2_5 = memory_instance.add(conversation_2_5, user_id=user_id)
    print(f"Add results: {results_2_5}")
    print_current_memory(memory_instance, "Movie Recommendation")

    # Step 8: Ask for details about the movie
    conversation_2_6 = [
        {"role": "user", "content": "Tell me a bit about 'The Big Heist'."},
        {"role": "assistant", "content": "'The Big Heist' is a fast-paced film about a group of unlikely thieves pulling off a daring bank robbery."}
    ]
    print("\n--- Asking for details about the movie ---")
    results_2_6 = memory_instance.add(conversation_2_6, user_id=user_id)
    print(f"Add results: {results_2_6}")
    print_current_memory(memory_instance, "Movie Details")

    # Step 9: User moves to a new location
    conversation_3_1 = [
        {"role": "user", "content": "I recently moved. I live in New York now."},
        {"role": "assistant", "content": "Understood. Updating your location to New York."} # This should trigger an UPDATE for location
    ]
    print("\n--- User moves to a new location ---")
    results_3_1 = memory_instance.add(conversation_3_1, user_id=user_id)
    print(f"Add results: {results_3_1}")
    print_current_memory(memory_instance, "Location Update to New York")

    # Step 10: User health update and snack recommendation based on health
    conversation_3_2 = [
        {"role": "user", "content": "Also, I recently got high cholesterol. Can you tell me where I should buy snacks?"},
        {"role": "assistant", "content": "With high cholesterol, it's best to look for low-fat, low-sodium snacks. You should consider health food stores or the organic section of supermarkets in New York for options like air-popped popcorn, nuts in moderation, or fruit."} # This should trigger an ADD/UPDATE related to snack preferences and health
    ]
    print("\n--- User health update and snack recommendation based on health ---")
    results_3_2 = memory_instance.add(conversation_3_2, user_id=user_id)
    print(f"Add results: {results_3_2}")
    print_current_memory(memory_instance, "Health Update & Healthy Snack Recommendation")

    # --- Test Case 1: Forgetting information from more than ten turns ago ---
    print("\n--- Testing Scenario 1: Forgetting older information ---")
    # Simulate forgetting information that is more than 10 turns old.
    # We'll assume "New Jersey" from Step 0 is now "forgotten" if a mechanism
    # like a sliding window or time-based decay were in place and not managed by explicit updates.
    # However, with current mem0, all memories are accessible unless explicitly deleted or superseded.
    # To *simulate* a loss of context, we can ask a question that relies on the old information
    # and observe if the LLM can still infer it, or if it needs to be explicitly re-introduced.

    # Let's ask about supermarkets in the *current* location (New York),
    # but frame it in a way that might implicitly recall the previous location if not handled well.
    print("\n--- Asking about supermarkets in the current location (New York) ---")
    # This query *should* leverage the "New York" update from Step 9.
    # If the system has lost the context of "New Jersey" and doesn't properly
    # infer that the user is now asking about the *new* location, it might be confused.
    # The key here is that the LLM should understand "near me" refers to the most recent location.
    conversation_s1_1 = [
        {"role": "user", "content": "Are there any good supermarkets near me?"},
        {"role": "assistant", "content": "Searching for supermarkets near you in New York..."} # Expected LLM response indicating it's using the current location.
    ]
    results_s1_1 = memory_instance.add(conversation_s1_1, user_id=user_id)
    print(f"Add results: {results_s1_1}")
    print_current_memory(memory_instance, "Supermarkets in New York (after forgetting old info simulation)")

    # Let's try to ask about movie theaters again, and see if it defaults to the old location or the new one.
    # The LLM should ideally infer "near me" refers to the current location, New York.
    print("\n--- Asking about movie theaters again, checking for current location context ---")
    conversation_s1_2 = [
        {"role": "user", "content": "What about movie theaters?"},
        {"role": "assistant", "content": "Searching for movie theaters near you in New York..."} # Expected LLM response indicating it's using the current location.
    ]
    results_s1_2 = memory_instance.add(conversation_s1_2, user_id=user_id)
    print(f"Add results: {results_s1_2}")
    print_current_memory(memory_instance, "Movie Theaters in New York (after forgetting old info simulation)")

    # --- Test Case 2: Incorrect information not being deleted ---
    print("\n--- Testing Scenario 2: Incorrect information leading to misdirection ---")
    # Let's inject some incorrect information and see if it pollutes subsequent reasoning.
    # We will add a memory that states a false preference.
    print("\n--- Injecting false information: User dislikes pizza ---")
    conversation_s2_1_incorrect = [
        {"role": "user", "content": "Just to be clear, I actually hate pizza."},
        {"role": "assistant", "content": "Understood. You dislike pizza. Noted."} # This is a false statement for the scenario.
    ]
    results_s2_1_incorrect = memory_instance.add(conversation_s2_1_incorrect, user_id=user_id)
    print(f"Add results: {results_s2_1_incorrect}")
    print_current_memory(memory_instance, "After Injecting False 'Dislikes Pizza'")

    # Now, let's ask a question where this false information might interfere.
    # For instance, asking for a food recommendation.
    print("\n--- Asking for a food recommendation, hoping it avoids pizza due to false info ---")
    # If the memory manager correctly handles the implicit "dislikes pizza" or the LLM
    # can contextualize that this is a new preference overriding older implicit ones,
    # it should recommend something else.
    # However, if the false information is prioritized or not contextualized, it could be problematic.
    conversation_s2_2_query = [
        {"role": "user", "content": "What's a good dinner option for tonight?"},
        {"role": "assistant", "content": "Considering you dislike pizza, how about trying a nice pasta dish or some grilled chicken? Both are popular choices."} # Expected LLM response avoiding pizza.
    ]
    results_s2_2_query = memory_instance.add(conversation_s2_2_query, user_id=user_id)
    print(f"Add results: {results_s2_2_query}")
    print_current_memory(memory_instance, "Food Recommendation After False Info")

    # Let's try to 'correct' the false information.
    print("\n--- Correcting the false information: User likes pizza ---")
    conversation_s2_3_correction = [
        {"role": "user", "content": "Actually, I was wrong. I do like pizza. I'd love to find a good pizza place in New York."},
        {"role": "assistant", "content": "My apologies! It's good to know you enjoy pizza. Let me find some great pizza places for you in New York."}
    ]
    results_s2_3_correction = memory_instance.add(conversation_s2_3_correction, user_id=user_id)
    print(f"Add results: {results_s2_3_correction}")
    print_current_memory(memory_instance, "After Correcting False 'Dislikes Pizza'")

    # Now, ask again for a dinner recommendation, and see if it now suggests pizza.
    print("\n--- Asking for food recommendation again, after correcting the false info ---")
    conversation_s2_4_query_corrected = [
        {"role": "user", "content": "What's a good dinner option for tonight?"},
        {"role": "assistant", "content": "Since you enjoy pizza, I can help you find some excellent pizza restaurants in New York. Would you like recommendations for a specific style, like Neapolitan or New York-style?"} # Expected LLM response suggesting pizza.
    ]
    results_s2_4_query_corrected = memory_instance.add(conversation_s2_4_query_corrected, user_id=user_id)
    print(f"Add results: {results_s2_4_query_corrected}")
    print_current_memory(memory_instance, "Food Recommendation After Corrected False Info")


    print("\n--- Final Memory State after all optimized interactions and tests ---")
    print_current_memory(memory_instance, "Final State")

# Run the optimized test scenario
run_optimized_test_scenario(m)