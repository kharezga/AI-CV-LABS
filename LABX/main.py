from pomegranate import *

random_events = Node(DiscreteDistribution({

    "break up": 0.2,
    "relative lost": 0.1,
    "financial problem": 0.3,
    "study pressure": 0.3,
    "no friends": 0.5,
    "serious illness": 0.2,

}), name="events")

#emotional shock dependant from random events
emotional_shock = Node(ConditionalProbabilityTable([

    ["break up", "relative lost", "emotional shock", 0.85],
    ["break up", "none", "emotional shock", 0.4],
    ["none", "relative lost", "emotional shock", 0.6],
    ["none", "none", "emotional shock", 0.05]
    # ["serious illness", "prior stroke", 0.95], # consider second option "["serious illness", "none" , "prior stroke"...

], [random_events.distribution]), name = "Sub reason")


# reason dependant from emotional shock and random events
reason = Node(ConditionalProbabilityTable([

    ["emotional shock", "no friends", "depression", 0.98],
    ["emotional shock", "none", "depression", 0.75],
    ["none", "no friends", "depression", 0.8],
    ["none", "none", "depression", 0.85],
    ["financial problem", "study pressure", "chronic stress", 0.95],
    ["financial problem", "none", "chronic stress", 0.8],
    ["none", "study pressure", "chronic stress", 0.6],
    ["none", "none", "chronic stress", 0.1],
    ["serious illness", "none", "prior stroke", 0.95],
    ["none", "none", "prior stroke", 0.2]

], [emotional_shock.distribution],[random_events.distribution]), name = "Reasons")

# insomnia dependant from reasons of suicide
insomnia = Node(ConditionalProbabilityTable([

    ["chronic stress", "insomnia", 0.95],
    ["none", "insomnia", 0.75]

], [reason.distribution]), name = "Insomnia")

suicide = Node(ConditionalProbabilityTable([

    ["depression", "chronic stress", "prior stroke",  0.8],
    ["depression", "chronic stress", "none",  0.95],
    ["depression", "none", "prior stroke",  0.65],
    ["depression", "none", "none",  0.75],
    ["none", "chronic stress", "prior stroke",  0.3],
    ["none", "chronic stress", "none",  0.4],
    ["none", "none", "prior stroke",  0.02],
    ["none", "none", "none",  0.3],

], [reason.distribution]), name = "Suicide")

def main():

# Create a Bayesian Network and add arguments
universe = BayesianNetwork()
universe.add_states(random_events, emotional_shock, reason, insomnia, suicide)

universe.add_edge(random_events, emotional_shock)
universe.add_edge(random_events, emotional_shock, reason)
universe.add_edge(reason, insomnia)
universe.add_edge(reason, suicide)

#Finalize model
universe.bake()


if __name__ == "__main__":
    main()
