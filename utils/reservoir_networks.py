from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
from sklearn.pipeline import Pipeline, FeatureUnion

def Vanilla_input2node(I2N_config):
    return InputToNode(
        hidden_layer_size=I2N_config["layer_size"], k_in=I2N_config["sparsity"], input_activation=I2N_config["activation"],
        input_scaling=I2N_config["scaling"], input_shift=I2N_config["shift"], bias_scaling=I2N_config["bias_scaling"],
        bias_shift=I2N_config["bias_shift"], random_state=I2N_config["random_seed"]
    )
    
def Vanilla_node2node(N2N_config):
    return NodeToNode(
        hidden_layer_size=N2N_config["layer_size"], sparsity=N2N_config["sparsity"], reservoir_activation=N2N_config["activation"],
        spectral_radius=N2N_config["spectral_radius"], leakage=N2N_config["leakage"], bidirectional=N2N_config["bidirectional"],
        random_state=N2N_config["random_seed"]
    )

def Sequential_block(I2N_config, N2N_config, blocks=1):
    input2node = Vanilla_input2node(I2N_config)
    pipeline = [('i2n', input2node)]
    for i in range(blocks):
        node2node = Vanilla_node2node(N2N_config)
        pipeline.append(('n2n' + str(i+1), node2node))
    return Pipeline(pipeline)

def Parallel_block(I2N_config, N2N_config, blocks=2):
    pipeline = []
    for i in range(blocks):
        block = Sequential_block(I2N_config, N2N_config)
        pipeline.append(('block_'+str(i+1), block))
    return FeatureUnion(pipeline)
    

def reservoir_network(I2N, N2N):
    return ESNRegressor(input_to_node=I2N, node_to_node=N2N)