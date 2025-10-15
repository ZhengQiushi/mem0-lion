import os
from mem0 import Memory
from faker import Faker  # Import the Faker library for generating fake data
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
            # "temperature": 0.2,
            # "top_p": 1.0
        }
    },
    "graph_store": {
        "provider": "memgraph",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "memgraph",
            "password": "xxx"
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
    }
}

m = Memory.from_config(config)
fake = Faker()

# Generate 50 different topics and add them as memories
num_topics = 1
all_topics = []

for i in range(num_topics):
    # Generate a random conversation related to a topic
    topic_category = fake.word().capitalize()
    user_message = f"I'm interested in learning about {topic_category}."
    assistant_response = f"That's a great topic! {topic_category} is known for its {fake.bs()} and {fake.catch_phrase()}."
    follow_up_user = f"Can you tell me more about its {fake.word()}?"
    follow_up_assistant = f"Certainly! The {fake.word()} of {topic_category} involves {fake.sentence()}."

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": follow_up_user},
        {"role": "assistant", "content": follow_up_assistant}
    ]

    print(json.dumps(messages, indent=4))
    # Store the generated topic for later querying
    all_topics.append(topic_category)

    # Add the messages as memories
    result = m.add(messages, user_id="alice", metadata={"topic": topic_category})
    print(json.dumps(result, indent=4))
    print(f"Added memory for topic: {topic_category}")

print("--------------------------------")
print(f"Successfully added {num_topics} memories.")
print("--------------------------------")

# Query for one of the generated topics
# Select a random topic from the ones we generated
query_topic = random.choice(all_topics)
print(f"Querying for memories related to: '{query_topic}'")

# Construct a query that is likely to retrieve memories about the selected topic
# We can use the topic name itself as the query, or a phrase related to it.
# For this example, we'll use a phrase that is likely to be in the conversation.
search_query = f"Tell me what you know about {query_topic}"

related_memories = m.search(query=search_query, user_id="alice")

print("\n--- Related Memories ---")
if related_memories:
    print(json.dumps(related_memories, indent=4))
    print("-" * 10)
else:
    print("No related memories found.")

# You can also get all memories to verify if they were stored (though this can be verbose)
# all_memories = m.get_all(user_id="alice")
# print("\n--- All Memories ---")
# for memory in all_memories:
#     print(f"Content: {memory['content']}")
#     print(f"Metadata: {memory['metadata']}")
#     print("-" * 10)