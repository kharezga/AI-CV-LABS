from pomegranate import *

# Define all Discrete Events


break_up = Node(DiscreteDistribution({

    "break up": 0.2,
    "no break up": 0.8

}), name="break up")

relative_lost = Node(DiscreteDistribution({

    "relative lost": 0.1,
    "no relative lost": 0.9

}), name="relative lost")

financial_problem = Node(DiscreteDistribution({

    "financial problem": 0.3,
    "no financial problem": 0.7

}), name="financial_problem")

study_pressure = Node(DiscreteDistribution({

    "study pressure": 0.3,
    "no study pressure": 0.7

}), name="study_pressure")

illness = Node(DiscreteDistribution({

    "illness": 0.3,
    "no illness": 0.7

}), name="illness")

no_fren = Node(DiscreteDistribution({

    "no friends": 0.5,
    "friends": 0.5

}), name="no_fren")

# emotional shock dependant from random events
emotional_shock = Node(ConditionalProbabilityTable([

    ["break up", "relative lost", "emotional shock", 0.85],
    ["break up", "relative lost", "no emotional shock", 0.15],
    ["break up", "no relative lost", "emotional shock", 0.4],
    ["break up", "no relative lost", "no emotional shock", 0.6],
    ["no break up", "relative lost", "emotional shock", 0.6],
    ["no break up", "relative lost", "no emotional shock", 0.4],
    ["no break up", "no relative lost", "emotional shock", 0.05],
    ["no break up", "no relative lost", "no emotional shock", 0.95]

], [break_up.distribution, relative_lost.distribution]), name="emotional_shock")

chronic_stress = Node(ConditionalProbabilityTable([

    ["financial problem", "study pressure", "stress", 0.95],
    ["financial problem", "study pressure", "no stress", 0.05],
    ["financial problem", "no study pressure", "stress", 0.8],
    ["financial problem", "no study pressure", "no stress", 0.2],
    ["no financial problem", "study pressure", "stress", 0.6],
    ["no financial problem", "study pressure", "no stress", 0.4],
    ["no financial problem", "no study pressure", "stress", 0.1],
    ["no financial problem", "no study pressure", "no stress", 0.9],

], [financial_problem.distribution, study_pressure.distribution]), name="chronic_stress")

prior_stroke = Node(ConditionalProbabilityTable([

    ["illness", "prior stroke", 0.95],
    ["illness", "no prior stroke", 0.05],
    ["no illness", "prior stroke", 0.2],
    ["no illness", "no prior stroke", 0.8]

], [illness.distribution]), name="prior_stroke")

insomnia = Node(ConditionalProbabilityTable([

    ["stress", "insomnia", 0.95],
    ["stress", "no insomnia", 0.05],
    ["no stress", "insomnia", 0.3],
    ["no stress", "no insomnia", 0.7],

], [illness.distribution]), name="insomnia")

depression = Node(ConditionalProbabilityTable([

    ["financial problem", "study pressure", "depression", 0.95],
    ["financial problem", "study pressure", "no depression", 0.05],
    ["financial problem", "no study pressure", "depression", 0.8],
    ["financial problem", "no study pressure", "no depression", 0.2],
    ["no financial problem", "study pressure", "depression", 0.6],
    ["no financial problem", "study pressure", "no depression", 0.4],
    ["no financial problem", "no study pressure", "depression", 0.1],
    ["no financial problem", "no study pressure", "no depression", 0.9]

], [no_fren.distribution, emotional_shock.distribution]), name="depression")

suicide = Node(ConditionalProbabilityTable([

    ["emotional_shock", "stress", "prior stroke", "suicide", 0.8],
    ["emotional_shock", "stress", "prior stroke", "no suicide", 0.2],
    ["emotional_shock", "stress", "no prior stroke", "suicide", 0.95],
    ["emotional_shock", "stress", "no prior stroke", "no suicide", 0.05],
    ["emotional_shock", "no stress", "prior stroke", "suicide", 0.65],
    ["emotional_shock", "no stress", "prior stroke", "no suicide", 0.35],
    ["emotional_shock", "no stress", "no prior stroke", "suicide", 0.75],
    ["emotional_shock", "no stress", "no prior stroke", "no suicide", 0.25],
    ["no emotional_shock", "stress", "prior stroke", "suicide", 0.3],
    ["no emotional_shock", "stress", "prior stroke", "no suicide", 0.7],
    ["no emotional_shock", "stress", "no prior stroke", "suicide", 0.4],
    ["no emotional_shock", "stress", "no prior stroke", "no suicide", 0.6],
    ["no emotional_shock", "no stress", "prior stroke", "suicide", 0.02],
    ["no emotional_shock", "no stress", "prior stroke", "no suicide", 0.98],
    ["no emotional_shock", "no stress", "no prior stroke", "suicide", 0.05],
    ["no emotional_shock", "no stress", "no prior stroke", "no suicide", 0.95],

], [emotional_shock.distribution, chronic_stress.distribution, prior_stroke]), name="suicide")

# Create a Bayesian Network and add arguments
universe = BayesianNetwork()
universe.add_states(break_up, relative_lost, financial_problem, financial_problem,
                    study_pressure, illness, emotional_shock, no_fren, chronic_stress, prior_stroke, suicide)

universe.add_edge(emotional_shock, break_up)
universe.add_edge(emotional_shock, relative_lost)

universe.add_edge(chronic_stress, study_pressure)
universe.add_edge(chronic_stress, financial_problem)
universe.add_edge(chronic_stress, study_pressure)

# might cause errors
universe.add_edge(insomnia, chronic_stress)

universe.add_edge(prior_stroke, illness)
universe.add_edge(depression, no_fren)
universe.add_edge(depression, emotional_shock)

universe.add_edge(suicide, emotional_shock)
universe.add_edge(suicide, prior_stroke)
universe.add_edge(suicide, chronic_stress)

# Finalize model
universe.bake()

getProbability = universe.probability([["break up", "no relative lost", "financial problem", "study pressure",
                                        "no illness", "friends", "emotional shock", "stress", "no prior stroke", "depression", "suicide"]])

print(getProbability)
