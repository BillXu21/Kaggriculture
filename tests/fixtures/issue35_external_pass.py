"""Tiny subprocess fixture for the asymmetric evaluation harness."""


def agent(observation, configuration):
    del configuration
    hands = observation.get("farms", [])[int(observation["player"])].get(
        "hands", [])
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in hands],
        "market": [],
    }
