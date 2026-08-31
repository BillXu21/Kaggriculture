"""One-argument callable fixture for the pinned dispatch contract."""


def agent(observation):
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in observation.farms[observation.player].hands],
        "market": [],
    }
