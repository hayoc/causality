import itertools

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

from causality.graph import draw_network


def create_model_and_inference():
    dep_df = pd.read_csv('dependencies.csv', sep=';')

    def connect(df, source, edgelist):
        source_df = df[df['Column2'] == source]
        for col in source_df.iloc[0, 3:len(source_df.columns)]:
            target_df = df[df['Column1'] == col]['Column2']
            if not target_df.empty:
                target = target_df.item()
                if not (target, source) in edgelist:
                    edgelist.append((source, target))
                    connect(df, target, edgelist)

    edges = []
    connect(dep_df, 'myproximus-usage', edges)
    edges = [(t[1], t[0]) for t in edges]

    nodes = set(itertools.chain.from_iterable(edges))
    nodes_df = dep_df.iloc[:, 1].to_frame()
    nodes_df = nodes_df[nodes_df['Column2'].isin(nodes)]

    nodes_df['0'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['1'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['2'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['3'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['4'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['5'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['6'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['7'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['8'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['9'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df['10'] = pd.DataFrame(data=np.random.randint(0, 2, size=64).T)
    nodes_df = nodes_df.set_index('Column2').transpose()

    model = BayesianModel()
    model.add_nodes_from(nodes)
    for edge in edges:
        try:
            model.add_edge(edge[0], edge[1])
        except:
            print('WARNING: tried to add edge which forms loop: ' + str(edge))

    model.fit(nodes_df, estimator=BayesianEstimator, prior_type="BDeu")
    # for cpd in model.get_cpds():
    #     print(cpd)

    draw_network(model.nodes(), model.edges(), {}, [])

    return model, VariableElimination(model)